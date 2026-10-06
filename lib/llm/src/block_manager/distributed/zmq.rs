// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;
use tmq::AsZmqSocket;

use super::*;
use utils::*;

use anyhow::Result;
use async_trait::async_trait;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tmq::{
    Context, Message, Multipart,
    publish::{Publish, publish},
    pull::{Pull, pull},
    push::{Push, push},
    subscribe::{Subscribe, subscribe},
};
use tokio::sync::{Mutex, oneshot};
use tokio_util::sync::CancellationToken;

use bincode;
use futures_util::{SinkExt, StreamExt};
use std::cmp::min;

struct PendingMessage {
    remaining_workers: usize,
    completion_indicator: Option<oneshot::Sender<()>>,
    // If true, collect one payload (bytes) from each worker reply.
    want_payload: bool,
    // Collected raw payloads (one per worker), if want_payload == true
    payloads: Option<Vec<Vec<u8>>>,
    // If set, receives the collected payloads instead of `completion_indicator`, and the
    // message is forgotten once every worker answered (see `broadcast_with_payloads`).
    payload_indicator: Option<oneshot::Sender<Vec<Vec<u8>>>>,
}

impl PendingMessage {
    fn with_payloads(num_workers: usize, payload_indicator: oneshot::Sender<Vec<Vec<u8>>>) -> Self {
        Self {
            remaining_workers: num_workers,
            completion_indicator: None,
            want_payload: true,
            payloads: Some(Vec::with_capacity(num_workers)),
            payload_indicator: Some(payload_indicator),
        }
    }

    /// Record one worker's answer: `payload` is the first data frame of a reply, or `None` for a
    /// bare ack. Returns true once every worker has answered.
    fn record_reply(&mut self, payload: Option<&[u8]>) -> bool {
        if self.want_payload
            && let (Some(payload), Some(payloads)) = (payload, self.payloads.as_mut())
        {
            payloads.push(payload.to_vec());
        }
        self.remaining_workers = self.remaining_workers.saturating_sub(1);
        self.remaining_workers == 0
    }
}

pub struct LeaderSockets {
    pub pub_socket: Publish,
    pub pub_url: String,
    pub ack_socket: Pull,
    pub ack_url: String,
}

pub fn new_leader_sockets(pub_url: &str, ack_url: &str) -> Result<LeaderSockets> {
    let context = Context::new();
    let pub_socket = publish(&context).bind(pub_url)?;
    let pub_url = pub_socket
        .get_socket()
        .get_last_endpoint()
        .unwrap()
        .unwrap();

    let ack_socket = pull(&context).bind(ack_url)?;
    let ack_url = ack_socket
        .get_socket()
        .get_last_endpoint()
        .unwrap()
        .unwrap();

    Ok(LeaderSockets {
        pub_socket,
        pub_url,
        ack_socket,
        ack_url,
    })
}

/// The ActiveMessageLeader is responsible for sending commands to all workers.
/// On the leader side, we use two sockets:
/// 1. A publish socket to send messages to all workers.
/// 2. A pull socket to receive ACKs from workers.
pub struct ZmqActiveMessageLeader {
    // Our socket to broadcast messages.
    pub_socket: Arc<Mutex<Publish>>,
    // Message ID counter. Used for ACKs
    message_id: Arc<Mutex<usize>>,
    // Map of currently pending messages (messages that haven't been ACKed by all workers).
    pending_messages: Arc<Mutex<HashMap<usize, PendingMessage>>>,
    // Number of workers we're waiting for.
    num_workers: Arc<usize>,
}

impl ZmqActiveMessageLeader {
    /// Handshake-first constructor: collects WorkerMetaData, broadcasts LeaderMetadata,
    /// waits for allocation ACKs, then runs the final ping loop.
    pub async fn new_with_handshake<F>(
        leader_sockets: LeaderSockets,
        num_workers: usize,
        overall_timeout: Duration,
        cancel_token: CancellationToken,
        make_leader_meta: F,
    ) -> Result<Self>
    where
        F: Fn(&[WorkerMetadata]) -> LeaderMetadata + Send + Sync + 'static,
    {
        let pub_socket = Arc::new(Mutex::new(leader_sockets.pub_socket));
        let pull_socket = leader_sockets.ack_socket;

        tracing::info!(
            "ZmqActiveMessageLeader: Bound to pub: {} and pull: {}",
            leader_sockets.pub_url,
            leader_sockets.ack_url
        );

        let pending_messages = Arc::new(Mutex::new(HashMap::new()));
        let pending_messages_clone = pending_messages.clone();
        CriticalTaskExecutionHandle::new(
            |ct| Self::pull_worker(pull_socket, pending_messages_clone, ct),
            cancel_token.clone(),
            "ZmqActiveMessageLeader: Pull worker",
        )?
        .detach();

        let this = Self {
            pub_socket,
            message_id: Arc::new(Mutex::new(0)),
            pending_messages,
            num_workers: Arc::new(num_workers),
        };

        let deadline = Instant::now() + overall_timeout;

        // 1) Collect KvbmWorkerData from ALL workers in a single round.
        // Keep rebroadcasting until we get exactly `num_workers` replies to the SAME broadcast.
        let workers_payloads: Vec<Vec<u8>> = loop {
            if Instant::now() >= deadline {
                return Err(anyhow::anyhow!(
                    "Handshake timed out (device-config collection)."
                ));
            }
            let remain = deadline.saturating_duration_since(Instant::now());
            let round_to = min(Duration::from_secs(2), remain);

            tracing::info!("Handshake: requesting worker device configs...");
            match this
                .broadcast_collect(
                    ZMQ_WORKER_METADATA_MESSAGE,
                    &[],
                    /* want_payload */ true,
                    round_to,
                )
                .await
            {
                Ok(payloads) if payloads.len() == num_workers => {
                    tracing::info!(
                        "Handshake: received {} worker metadata replies in this round.",
                        payloads.len()
                    );
                    break payloads;
                }
                Ok(payloads) => {
                    tracing::warn!(
                        "Handshake: got {} / {} worker metadata replies; rebroadcasting...",
                        payloads.len(),
                        num_workers
                    );
                    continue;
                }
                Err(e) => {
                    tracing::debug!(
                        "Handshake: worker metadata round timed out/failed: {e}; retrying..."
                    );
                    continue;
                }
            }
        };

        let mut workers: Vec<WorkerMetadata> = Vec::with_capacity(workers_payloads.len());

        for payload in workers_payloads {
            let worker: WorkerMetadata =
                bincode::serde::decode_from_slice(&payload, bincode::config::standard())?.0;
            workers.push(worker);
        }

        // 2) Compute & broadcast LeaderMetadata; wait for ALL acks in the SAME round.
        let leader_meta = make_leader_meta(&workers);
        let leader_meta_bytes =
            bincode::serde::encode_to_vec(&leader_meta, bincode::config::standard())?;

        loop {
            if Instant::now() >= deadline {
                return Err(anyhow::anyhow!(
                    "Handshake timed out (allocation-config broadcast)."
                ));
            }
            let remain = deadline.saturating_duration_since(Instant::now());
            let round_to = min(Duration::from_secs(2), remain);

            tracing::info!("Handshake: broadcasting allocation config to workers...");
            match this
                .broadcast_collect(
                    ZMQ_LEADER_METADATA_MESSAGE,
                    std::slice::from_ref(&leader_meta_bytes),
                    /* want_payload */ false,
                    round_to,
                )
                .await
            {
                Ok(_) => {
                    // Success: all workers acked in this round.
                    tracing::info!("Handshake: all workers acked allocation config.");
                    break;
                }
                Err(e) => {
                    tracing::warn!(
                        "Handshake: allocation-config round incomplete: {e}; rebroadcasting..."
                    );
                    continue;
                }
            }
        }

        // 3) Final readiness ping loop (workers only ACK after allocation ready)
        let ping_deadline = deadline;
        loop {
            if Instant::now() >= ping_deadline {
                return Err(anyhow::anyhow!(
                    "Timed out waiting for ping readiness after handshake."
                ));
            }
            tracing::debug!("Handshake: final readiness ping...");
            let ping = this.broadcast(ZMQ_PING_MESSAGE, vec![]).await?;
            tokio::select! {
                _ = ping => break,
                _ = tokio::time::sleep(Duration::from_millis(500)) => continue,
                _ = cancel_token.cancelled() => return Err(anyhow::anyhow!("Startup canceled")),
            }
        }

        Ok(this)
    }

    /// Broadcast a message to all workers.
    /// Returns a receiver that will be notified when all workers have ACKed the message.
    pub async fn broadcast(
        &self,
        function: &str,
        data: Vec<Vec<u8>>,
    ) -> Result<oneshot::Receiver<()>> {
        // Generate a unique id.
        let id = {
            let mut id = self.message_id.lock().await;
            *id += 1;
            *id
        };

        let (completion_indicator, completion_receiver) = oneshot::channel();

        let pending_message = PendingMessage {
            // We start with the number of workers we're waiting for.
            remaining_workers: *self.num_workers,
            completion_indicator: Some(completion_indicator),
            want_payload: false,
            payloads: None,
            payload_indicator: None,
        };

        // Add the message to the pending messages map.
        self.pending_messages
            .lock()
            .await
            .insert(id, pending_message);

        // id, function, data
        let mut message: VecDeque<Message> = VecDeque::with_capacity(data.len() + 2);
        message.push_back(id.to_be_bytes().as_slice().into());
        message.push_back(function.into());
        for data in data {
            message.push_back(data.into());
        }

        tracing::debug!(
            "ZmqActiveMessageLeader: Broadcasting message with id: {}",
            id
        );
        self.pub_socket
            .lock()
            .await
            .send(Multipart(message))
            .await?;

        Ok(completion_receiver)
    }

    /// Broadcast a message and collect the first payload frame of each worker's reply.
    /// The receiver yields the payloads once every worker has answered; a worker that only
    /// acked contributes no payload, so callers can detect it.
    pub async fn broadcast_with_payloads(
        &self,
        function: &str,
        data: Vec<Vec<u8>>,
    ) -> Result<oneshot::Receiver<Vec<Vec<u8>>>> {
        let id = {
            let mut id = self.message_id.lock().await;
            *id += 1;
            *id
        };

        let (payload_indicator, payload_receiver) = oneshot::channel();
        self.pending_messages.lock().await.insert(
            id,
            PendingMessage::with_payloads(*self.num_workers, payload_indicator),
        );

        // id, function, data
        let mut message: VecDeque<Message> = VecDeque::with_capacity(data.len() + 2);
        message.push_back(id.to_be_bytes().as_slice().into());
        message.push_back(function.into());
        for data in data {
            message.push_back(data.into());
        }

        tracing::debug!(
            "ZmqActiveMessageLeader: Broadcasting message with id {} (collecting replies)",
            id
        );
        if let Err(error) = self.pub_socket.lock().await.send(Multipart(message)).await {
            self.pending_messages.lock().await.remove(&id);
            return Err(error.into());
        }

        Ok(payload_receiver)
    }

    /// Generic broadcast that can collect one reply payload from each worker.
    /// - `function`: handler name on workers
    /// - `data_frames`: optional extra frames after [id, function]
    /// - `want_payload`: if true, expects replies shaped as [id, function, payload]
    ///   Returns payloads (empty if want_payload == false).
    pub async fn broadcast_collect(
        &self,
        function: &str,
        data_frames: &[Vec<u8>],
        want_payload: bool,
        timeout: Duration,
    ) -> Result<Vec<Vec<u8>>> {
        // Generate a unique id.
        let id = {
            let mut id = self.message_id.lock().await;
            *id += 1;
            *id
        };

        let (completion_indicator, completion_receiver) = oneshot::channel();
        let pending_message = PendingMessage {
            remaining_workers: *self.num_workers,
            completion_indicator: Some(completion_indicator),
            want_payload,
            payloads: want_payload.then(|| Vec::with_capacity(*self.num_workers)),
            payload_indicator: None,
        };
        self.pending_messages
            .lock()
            .await
            .insert(id, pending_message);

        // Build message: [id, function, ...data]
        let mut message: VecDeque<Message> = VecDeque::with_capacity(2 + data_frames.len());
        message.push_back(id.to_be_bytes().as_slice().into());
        message.push_back(function.into());
        for df in data_frames {
            message.push_back(df.clone().into());
        }
        self.pub_socket
            .lock()
            .await
            .send(Multipart(message))
            .await?;

        // Await all replies or timeout.
        tokio::select! {
            _ = completion_receiver => { /* done */ }
            _ = tokio::time::sleep(timeout) => {
                let mut map = self.pending_messages.lock().await;
                map.remove(&id);
                return Err(anyhow::anyhow!("Timed out waiting for '{}' responses", function));
            }
        }

        // Extract payloads (if any).
        let mut map = self.pending_messages.lock().await;
        let entry = map
            .remove(&id)
            .ok_or_else(|| anyhow::anyhow!("pending entry missing"))?;
        Ok(entry.payloads.unwrap_or_default())
    }

    async fn pull_worker(
        mut pull_socket: Pull,
        pending_messages: Arc<Mutex<HashMap<usize, PendingMessage>>>,
        cancel_token: CancellationToken,
    ) -> Result<()> {
        loop {
            tokio::select! {
                Some(Ok(message)) = pull_socket.next() => {
                if message.is_empty() {
                    tracing::error!("Leader PULL: empty message");
                    continue;
                }
                let arr: [u8; std::mem::size_of::<usize>()] = (*message[0]).try_into()?;
                let id = usize::from_be_bytes(arr);

                let mut map = pending_messages.lock().await;

                let Some(pm) = map.get_mut(&id) else {
                    // Late reply for a round we've already collected/removed.
                    tracing::debug!("Leader PULL: late/unknown id {}", id);
                    continue;
                };

                // payload reply or pure ACK?
                let payload = (message.len() >= 3).then(|| &*message[2]);
                let complete = pm.record_reply(payload);

                tracing::debug!(
                    "Leader PULL: got {} for id {} (remaining={})",
                    if message.len()==1 { "ACK" } else { "REPLY" }, id, pm.remaining_workers
                );

                if !complete {
                    continue;
                }
                if let Some(tx) = pm.payload_indicator.take() {
                    // Payload collectors are done with the message once every worker answered.
                    let payloads = pm.payloads.take().unwrap_or_default();
                    map.remove(&id);
                    let _ = tx.send(payloads);
                } else if let Some(tx) = pm.completion_indicator.take() {
                    // IMPORTANT: do NOT remove here; just notify completion.
                    let _ = tx.send(());
                }
            }
                _ = cancel_token.cancelled() => {
                    tracing::info!("ZmqActiveMessageLeader: Pull worker cancelled.");
                    break;
                }
            }
        }
        tracing::info!("ZmqActiveMessageLeader: Pull worker exiting.");
        Ok(())
    }
}

/// A message handle is used to track a message.
/// It contains a way to ACK the message, as well as the data.
pub struct MessageHandle {
    pub message_id: usize,
    function: String,
    pub data: Vec<Vec<u8>>,
    pub push_handle: Arc<Mutex<Push>>,
    acked: bool,
}

impl MessageHandle {
    pub fn new(message: Multipart, push_handle: Arc<Mutex<Push>>) -> Result<Self> {
        // We always need at least the message id and the function name.
        if message.len() < 2 {
            return Err(anyhow::anyhow!(
                "Received message with unexpected length: {:?}",
                message.len()
            ));
        }
        let arr: [u8; std::mem::size_of::<usize>()] = (*message[0]).try_into()?;
        let id = usize::from_be_bytes(arr);
        let function = message[1]
            .as_str()
            .ok_or(anyhow::anyhow!("Unable to parse function name."))?
            .to_string();

        // Skip the message id and function name: Everything else is data.
        let data = message.into_iter().skip(2).map(|m| (*m).to_vec()).collect();

        Ok(Self {
            message_id: id,
            function,
            data,
            push_handle,
            acked: false,
        })
    }

    /// ACK the message, which notifies the leader.
    pub async fn ack(&mut self) -> Result<()> {
        // We can only ACK once.
        if self.acked {
            return Err(anyhow::anyhow!("Message was already acked!"));
        }

        self.acked = true;

        let id = self.message_id;
        let mut message = VecDeque::with_capacity(1);
        message.push_back(id.to_be_bytes().as_slice().into());
        let message = Multipart(message);
        self.push_handle.lock().await.send(message).await?;
        tracing::debug!("ZmqActiveMessageWorker: ACKed message with id: {}", id);
        Ok(())
    }

    /// Reply to the leader with arbitrary payload frames and mark as acked.
    /// Frames shape: [id, function, payload_0, payload_1, ...]
    pub async fn reply(
        &mut self,
        function: &str,
        payload_frames: &[Vec<u8>],
    ) -> anyhow::Result<()> {
        let mut frames: std::collections::VecDeque<tmq::Message> =
            std::collections::VecDeque::with_capacity(2 + payload_frames.len());
        frames.push_back(self.message_id.to_be_bytes().as_slice().into());
        frames.push_back(function.into());
        for p in payload_frames {
            frames.push_back(p.clone().into());
        }
        self.push_handle
            .lock()
            .await
            .send(tmq::Multipart(frames))
            .await?;
        // Mark as acked so Drop won't panic; leader treats the reply as the "ack".
        self.acked = true;
        Ok(())
    }

    /// Mark this message as handled locally without sending an ACK/reply.
    /// Use when intentionally ignoring a message (e.g. ping before readiness).
    pub fn mark_handled(&mut self) {
        self.acked = true;
    }
}

/// We must always ACK a message.
/// Panic if we don't.
impl Drop for MessageHandle {
    fn drop(&mut self) {
        if !self.acked {
            panic!("Message was not acked!");
        }
    }
}

/// A handler is responsible for handling a message.
/// We have to use this instead of AsyncFn because AsyncFn isn't dyn compatible.
#[async_trait]
pub trait Handler: Send + Sync {
    async fn handle(&self, message: MessageHandle) -> Result<()>;
}

type MessageHandlers = HashMap<String, Arc<dyn Handler>>;

/// The ActiveMessageWorker receives commands from the leader, and ACKs them.
pub struct ZmqActiveMessageWorker {}

impl ZmqActiveMessageWorker {
    pub fn new(
        sub_url: &str,
        push_url: &str,
        message_handlers: MessageHandlers,
        cancel_token: CancellationToken,
    ) -> Result<Self> {
        let context = Context::new();

        let sub_socket = subscribe(&context)
            .connect(sub_url)?
            .subscribe("".as_bytes())?;
        let push_socket = Arc::new(Mutex::new(push(&context).connect(push_url)?));

        tracing::info!(
            "ZmqActiveMessageWorker: Bound to sub: {} and push: {}",
            sub_url,
            push_url
        );

        let message_handlers = Arc::new(message_handlers);

        CriticalTaskExecutionHandle::new(
            |cancel_token| {
                Self::sub_worker(sub_socket, push_socket, message_handlers, cancel_token)
            },
            cancel_token,
            "ZmqActiveMessageWorker: Sub worker",
        )?
        .detach();

        Ok(Self {})
    }

    async fn sub_worker(
        mut sub_socket: Subscribe,
        push_socket: Arc<Mutex<Push>>,
        message_handlers: Arc<MessageHandlers>,
        cancel_token: CancellationToken,
    ) -> Result<()> {
        loop {
            tokio::select! {
                Some(Ok(message)) = sub_socket.next() => {
                    if message.len() < 2 {
                        tracing::error!(
                            "Received message with unexpected length: {:?}",
                            message.len()
                        );
                        continue;
                    }

                    // Try to parse our message.
                    let message_handle = MessageHandle::new(message, push_socket.clone())?;

                    // Check if the function name is registered.
                    // TODO: We may want to make this dynamic, and expose a function
                    // to dynamically add/remove handlers.
                    if let Some(handler) = message_handlers.get(&message_handle.function) {
                        tracing::debug!(
                            "ZmqActiveMessageWorker: Handling message with id: {} for function: {}",
                            message_handle.message_id,
                            message_handle.function
                        );
                        let handler_clone = handler.clone();
                        let handle_text = format!("ZmqActiveMessageWorker: Handler for function: {}", message_handle.function);
                        CriticalTaskExecutionHandle::new(
                            move |_| async move { handler_clone.handle(message_handle).await },
                            cancel_token.clone(),
                            handle_text.as_str(),
                        )?
                        .detach();
                    } else {
                        tracing::error!("No handler found for function: {}", message_handle.function);
                    }
                }
                _ = cancel_token.cancelled() => {
                    break;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_message_collects_one_payload_per_worker() {
        let (tx, _rx) = oneshot::channel();
        let mut pending = PendingMessage::with_payloads(2, tx);
        assert!(!pending.record_reply(Some(b"first")));
        assert!(pending.record_reply(Some(b"second")));
        assert_eq!(
            pending.payloads.as_deref(),
            Some(&[b"first".to_vec(), b"second".to_vec()][..])
        );
    }

    #[test]
    fn bare_acks_complete_without_payloads() {
        let (tx, _rx) = oneshot::channel();
        let mut pending = PendingMessage::with_payloads(2, tx);
        assert!(!pending.record_reply(None));
        assert!(pending.record_reply(Some(b"only")));
        assert_eq!(pending.payloads.as_deref(), Some(&[b"only".to_vec()][..]));
    }

    #[test]
    fn payloads_are_ignored_when_not_wanted() {
        let (tx, _rx) = oneshot::channel();
        let mut pending = PendingMessage {
            remaining_workers: 1,
            completion_indicator: Some(tx),
            want_payload: false,
            payloads: None,
            payload_indicator: None,
        };
        assert!(pending.record_reply(Some(b"ignored")));
        assert!(pending.payloads.is_none());
        // Extra replies never underflow the counter.
        assert!(pending.record_reply(None));
    }

    #[tokio::test]
    async fn pull_worker_delivers_reply_payloads_and_forgets_the_message() {
        let context = Context::new();
        let pull_socket = pull(&context).bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = pull_socket
            .get_socket()
            .get_last_endpoint()
            .unwrap()
            .unwrap();
        let mut push_socket = push(&context).connect(&endpoint).unwrap();

        let pending = Arc::new(Mutex::new(HashMap::new()));
        let (tx, rx) = oneshot::channel();
        pending
            .lock()
            .await
            .insert(7usize, PendingMessage::with_payloads(2, tx));
        let cancel = CancellationToken::new();
        let worker = tokio::spawn(ZmqActiveMessageLeader::pull_worker(
            pull_socket,
            pending.clone(),
            cancel.clone(),
        ));

        for payload in [&b"one"[..], &b"two"[..]] {
            let mut frames: VecDeque<Message> = VecDeque::new();
            frames.push_back(7usize.to_be_bytes().as_slice().into());
            frames.push_back("transfer_blocks".into());
            frames.push_back(payload.into());
            push_socket.send(Multipart(frames)).await.unwrap();
        }

        let payloads = tokio::time::timeout(Duration::from_secs(10), rx)
            .await
            .expect("payloads were not delivered")
            .unwrap();
        assert_eq!(payloads, vec![b"one".to_vec(), b"two".to_vec()]);
        assert!(!pending.lock().await.contains_key(&7));

        cancel.cancel();
        worker.await.unwrap().unwrap();
    }
}
