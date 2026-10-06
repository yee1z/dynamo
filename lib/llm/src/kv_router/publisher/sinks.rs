// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::sync::Arc;

use anyhow::Result;

use dynamo_kv_router::RouterEventSink;
use dynamo_kv_router::indexer::LocalKvIndexer;
use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData, RouterEvent, StorageTier};
use dynamo_runtime::transports::event_plane::EventPublisher;
use dynamo_runtime::transports::nats::NatsQueue;

use crate::kv_router::KV_EVENT_SUBJECT;

pub(super) struct EventPlanePublisher(pub(super) EventPublisher);

impl RouterEventSink for EventPlanePublisher {
    fn publish_event(&self, event: &RouterEvent) -> impl Future<Output = Result<()>> + Send {
        self.0.publish(event)
    }
}

pub(super) struct JetStreamPublisher(pub(super) NatsQueue);

impl RouterEventSink for JetStreamPublisher {
    fn publish_event(&self, event: &RouterEvent) -> impl Future<Output = Result<()>> + Send {
        NatsQueue::publish_event(&self.0, KV_EVENT_SUBJECT, event)
    }
}

fn publish_trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("DYN_PHASE_D_STATE_TRACE").is_some())
}

/// Worker-side identity and time of one published KV event. Joined with the
/// router's `DYN_PHASE_D_STATE_EVENT` on (worker_id, dp_rank, storage_tier,
/// event_id); first/last external hashes guard the join.
pub(super) fn publish_trace(
    worker_id: u64,
    storage_tier: StorageTier,
    event: &KvCacheEvent,
    worker_publish_ns: u64,
) -> serde_json::Value {
    let (op, hashes): (&str, Vec<u64>) = match &event.data {
        KvCacheEventData::Stored(store) => (
            "stored",
            store
                .blocks
                .iter()
                .map(|block| block.block_hash.0)
                .collect(),
        ),
        KvCacheEventData::Removed(remove) => (
            "removed",
            remove.block_hashes.iter().map(|hash| hash.0).collect(),
        ),
        KvCacheEventData::Cleared => ("cleared", Vec::new()),
    };
    serde_json::json!({
        "schema": 1,
        "worker_publish_ns": worker_publish_ns,
        "worker_id": worker_id,
        "dp_rank": event.dp_rank,
        "event_id": event.event_id,
        "storage_tier": format!("{storage_tier:?}"),
        "op": op,
        "blocks": hashes.len(),
        "first_hash": hashes.first(),
        "last_hash": hashes.last(),
    })
}

pub(super) async fn emit<P: RouterEventSink>(
    publisher: &P,
    local_indexer: &Option<Arc<LocalKvIndexer>>,
    worker_id: u64,
    storage_tier: StorageTier,
    event: KvCacheEvent,
) {
    if publish_trace_enabled() {
        tracing::info!(
            "DYN_PHASE_D_KV_PUBLISH {}",
            publish_trace(
                worker_id,
                storage_tier,
                &event,
                crate::kv_router::monotonic_ns()
            )
        );
    }
    let router_event = RouterEvent::with_storage_tier(worker_id, event, storage_tier);
    if let Some(indexer) = local_indexer
        && let Err(e) = indexer.apply_event_with_buffer(router_event.clone()).await
    {
        tracing::warn!(worker_id, error = %e, "Failed to apply event to local indexer");
    }
    if let Err(e) = publisher.publish_event(&router_event).await {
        tracing::error!(worker_id, error = %e, "Failed to publish event");
    }
}

#[cfg(test)]
mod publish_trace_tests {
    use super::*;
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheRemoveData, KvCacheStoreData, KvCacheStoredBlockData,
        LocalBlockHash,
    };

    #[test]
    fn publish_trace_identifies_event_and_blocks() {
        let event = KvCacheEvent {
            event_id: 9,
            dp_rank: 1,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![
                    KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(5),
                        tokens_hash: LocalBlockHash(50),
                        mm_extra_info: None,
                    },
                    KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(6),
                        tokens_hash: LocalBlockHash(60),
                        mm_extra_info: None,
                    },
                ],
            }),
        };
        let trace = publish_trace(3, StorageTier::Device, &event, 1234);
        assert_eq!(trace["worker_publish_ns"], 1234);
        assert_eq!(trace["worker_id"], 3);
        assert_eq!(trace["dp_rank"], 1);
        assert_eq!(trace["event_id"], 9);
        assert_eq!(trace["storage_tier"], "Device");
        assert_eq!(trace["op"], "stored");
        assert_eq!(trace["blocks"], 2);
        assert_eq!(trace["first_hash"], 5);
        assert_eq!(trace["last_hash"], 6);
    }

    #[test]
    fn publish_trace_handles_removed_and_cleared() {
        let removed = KvCacheEvent {
            event_id: 10,
            dp_rank: 0,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(5)],
            }),
        };
        let trace = publish_trace(3, StorageTier::HostPinned, &removed, 1);
        assert_eq!(trace["op"], "removed");
        assert_eq!(trace["blocks"], 1);
        let cleared = KvCacheEvent {
            event_id: 11,
            dp_rank: 0,
            data: KvCacheEventData::Cleared,
        };
        let trace = publish_trace(3, StorageTier::Device, &cleared, 1);
        assert_eq!(trace["op"], "cleared");
        assert_eq!(trace["blocks"], 0);
        assert!(trace["first_hash"].is_null());
    }
}
