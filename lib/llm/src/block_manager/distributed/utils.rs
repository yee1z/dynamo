// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_getters::Getters;
use serde::{Deserialize, Serialize};

use crate::block_manager::connector::protocol::LeaderTransferRequest;

pub const ZMQ_PING_MESSAGE: &str = "ping";
pub const ZMQ_WORKER_METADATA_MESSAGE: &str = "worker_metadata";
pub const ZMQ_LEADER_METADATA_MESSAGE: &str = "leader_metadata";
pub const ZMQ_TRANSFER_BLOCKS_MESSAGE: &str = "transfer_blocks";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerMetadata {
    pub num_device_blocks: usize,
    pub bytes_per_block: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderMetadata {
    pub num_host_blocks: usize,
    pub num_disk_blocks: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Copy)]
pub enum BlockTransferPool {
    Device,
    Host,
    Disk,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ConnectorTransferType {
    Store,
    Load,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConnectorRequestLeader {
    pub req_id: String,
    pub txn_id: u64,
    pub transfer_type: ConnectorTransferType,
}

#[derive(Serialize, Deserialize, Debug, Getters, Clone)]
pub struct BlockTransferRequest {
    pub from_pool: BlockTransferPool,
    pub to_pool: BlockTransferPool,
    pub blocks: Vec<(usize, usize)>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub connector_req: Option<LeaderTransferRequest>,
}

impl BlockTransferRequest {
    #[allow(dead_code)]
    pub fn new(
        from_pool: BlockTransferPool,
        to_pool: BlockTransferPool,
        blocks: Vec<(usize, usize)>,
    ) -> Self {
        Self {
            from_pool,
            to_pool,
            blocks,
            connector_req: None,
        }
    }

    pub fn new_with_trigger_id(
        from_pool: BlockTransferPool,
        to_pool: BlockTransferPool,
        blocks: Vec<(usize, usize)>,
        connector_req: LeaderTransferRequest,
    ) -> Self {
        Self {
            from_pool,
            to_pool,
            blocks,
            connector_req: Some(connector_req),
        }
    }
}

/// One worker's result for a `ZMQ_TRANSFER_BLOCKS_MESSAGE`, sent as the payload of its reply.
/// A bare ack would make a failed transfer look complete to the leader.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferOutcome {
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl TransferOutcome {
    pub fn from_result(result: &anyhow::Result<()>) -> Self {
        match result {
            Ok(()) => Self {
                ok: true,
                error: None,
            },
            Err(error) => Self {
                ok: false,
                error: Some(format!("{error:#}")),
            },
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        // A struct of a bool and an optional string always serializes.
        serde_json::to_vec(self)
            .unwrap_or_else(|_| br#"{"ok":false,"error":"unencodable transfer outcome"}"#.to_vec())
    }
}

/// Fold the replies of all workers into one result. A missing or unreadable reply counts as a
/// failure, so a worker that only acked can never make a transfer look successful.
pub fn transfer_outcomes_result(payloads: &[Vec<u8>], num_workers: usize) -> anyhow::Result<()> {
    let mut failures = Vec::new();
    if payloads.len() < num_workers {
        failures.push(format!(
            "{} of {num_workers} workers replied without a transfer result",
            num_workers - payloads.len()
        ));
    }
    for payload in payloads {
        match serde_json::from_slice::<TransferOutcome>(payload) {
            Ok(outcome) if outcome.ok => {}
            Ok(outcome) => failures.push(
                outcome
                    .error
                    .unwrap_or_else(|| "worker reported a failed transfer".to_string()),
            ),
            Err(error) => failures.push(format!("unreadable transfer result: {error}")),
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "block transfer failed on a worker: {}",
            failures.join("; ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok() -> Vec<u8> {
        TransferOutcome::from_result(&Ok(())).encode()
    }

    fn failed(message: &str) -> Vec<u8> {
        TransferOutcome::from_result(&Err(anyhow::anyhow!("{message}"))).encode()
    }

    #[test]
    fn outcome_round_trips_through_its_payload() {
        let outcome = TransferOutcome::from_result(&Err(
            anyhow::anyhow!("no potential backend").context("Failed to write to blocks")
        ));
        assert!(!outcome.ok);
        let decoded: TransferOutcome = serde_json::from_slice(&outcome.encode()).unwrap();
        assert_eq!(decoded, outcome);
        let message = decoded.error.unwrap();
        assert!(message.contains("Failed to write to blocks"), "{message}");
        assert!(message.contains("no potential backend"), "{message}");

        let success: TransferOutcome = serde_json::from_slice(&ok()).unwrap();
        assert_eq!(
            success,
            TransferOutcome {
                ok: true,
                error: None
            }
        );
    }

    #[test]
    fn all_workers_ok_is_ok() {
        assert!(transfer_outcomes_result(&[ok()], 1).is_ok());
        assert!(transfer_outcomes_result(&[ok(), ok()], 2).is_ok());
    }

    #[test]
    fn any_failed_worker_fails_the_transfer() {
        let error = transfer_outcomes_result(&[ok(), failed("disk read failed")], 2).unwrap_err();
        assert!(format!("{error:#}").contains("disk read failed"));
    }

    #[test]
    fn missing_or_unreadable_replies_fail_the_transfer() {
        // A worker that only acked contributes no payload.
        assert!(transfer_outcomes_result(&[ok()], 2).is_err());
        assert!(transfer_outcomes_result(&[], 1).is_err());
        assert!(transfer_outcomes_result(&[b"not json".to_vec()], 1).is_err());
    }
}
