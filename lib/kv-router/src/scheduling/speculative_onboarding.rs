// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fail-closed qualification for M2 speculative-onboarding dry-run notices.
//!
//! This module carries metadata only. It never reserves blocks or starts transfers.

use std::sync::OnceLock;

use dynamo_tokens::SequenceHash;
use serde::{Deserialize, Serialize};

use crate::protocols::{DpRank, WorkerId, WorkerWithDpRank};

use super::{
    CacheIdentity, LowerTierFallbackReason, LowerTierStateStatus, LowerTierStateView,
    SelectedWorkerTierSnapshot,
};

pub const DYN_M2_DRY_RUN_NOTICE: &str = "DYN_M2_DRY_RUN_NOTICE";
pub const M2_NOTICE_EXTRA_ARGS_KEY: &str = "m2_speculative_onboarding_notice";
pub const M2_NOTICE_SCHEMA: u16 = 1;

pub fn speculative_onboarding_dry_run_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(DYN_M2_DRY_RUN_NOTICE)
            .ok()
            .is_some_and(|value| {
                matches!(
                    value.to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeculativeCacheIdentity {
    pub model: String,
    pub salt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<String>,
}

impl From<&CacheIdentity> for SpeculativeCacheIdentity {
    fn from(identity: &CacheIdentity) -> Self {
        Self {
            model: identity.model.clone(),
            salt: identity.salt.clone(),
            adapter: identity.adapter.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeculativeOnboardingPolicy {
    DryRun,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeculativeOnboardingNotice {
    pub schema: u16,
    pub request_id: String,
    pub worker_id: WorkerId,
    pub dp_rank: DpRank,
    pub identity: SpeculativeCacheIdentity,
    pub worker_epoch: u64,
    pub state_version: u64,
    pub state_age_ms: u64,
    pub prefix_start_block: u32,
    pub block_hashes: Vec<SequenceHash>,
    pub predicted_disk_blocks: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicted_queue_window_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_stage_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deadline_budget_ms: Option<u64>,
    pub policy: SpeculativeOnboardingPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculativeOnboardingRejectReason {
    QueryOnly,
    StateUnknown,
    StateStale,
    StateConflict,
    IdentityMismatch,
    MissingStateWatermark,
    UnsupportedDpTopology,
    NoDiskBlocks,
    MissingBlockHashes,
    BlockHashBounds,
    NonEmptyCacheSalt,
    InvalidExtraArgs,
    ConflictingNotice,
    DispatchTargetMismatch,
}

impl SpeculativeOnboardingRejectReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::QueryOnly => "query_only",
            Self::StateUnknown => "state_unknown",
            Self::StateStale => "state_stale",
            Self::StateConflict => "state_conflict",
            Self::IdentityMismatch => "identity_mismatch",
            Self::MissingStateWatermark => "missing_state_watermark",
            Self::UnsupportedDpTopology => "unsupported_dp_topology",
            Self::NoDiskBlocks => "no_disk_blocks",
            Self::MissingBlockHashes => "missing_block_hashes",
            Self::BlockHashBounds => "block_hash_bounds",
            Self::NonEmptyCacheSalt => "non_empty_cache_salt",
            Self::InvalidExtraArgs => "invalid_extra_args",
            Self::ConflictingNotice => "conflicting_notice",
            Self::DispatchTargetMismatch => "dispatch_target_mismatch",
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn qualify_speculative_onboarding_notice(
    request_id: Option<&str>,
    worker: WorkerWithDpRank,
    identity: &CacheIdentity,
    state: &LowerTierStateView,
    tiers: &SelectedWorkerTierSnapshot,
    sequence_hashes: Option<&[SequenceHash]>,
    isl_tokens: usize,
    block_size: u32,
) -> Result<SpeculativeOnboardingNotice, SpeculativeOnboardingRejectReason> {
    let request_id = request_id.ok_or(SpeculativeOnboardingRejectReason::QueryOnly)?;

    match state.status {
        LowerTierStateStatus::Known => {}
        LowerTierStateStatus::Unknown
            if state.fallback_reason == LowerTierFallbackReason::IdentityMismatch =>
        {
            return Err(SpeculativeOnboardingRejectReason::IdentityMismatch);
        }
        LowerTierStateStatus::Unknown => {
            return Err(SpeculativeOnboardingRejectReason::StateUnknown);
        }
        LowerTierStateStatus::Stale => {
            return Err(SpeculativeOnboardingRejectReason::StateStale);
        }
        LowerTierStateStatus::Conflict => {
            return Err(SpeculativeOnboardingRejectReason::StateConflict);
        }
    }

    let worker_epoch = state
        .worker_epoch
        .ok_or(SpeculativeOnboardingRejectReason::MissingStateWatermark)?;
    let state_version = state
        .version
        .ok_or(SpeculativeOnboardingRejectReason::MissingStateWatermark)?;
    let state_age_ms = u64::try_from(
        state
            .age
            .ok_or(SpeculativeOnboardingRejectReason::MissingStateWatermark)?
            .as_millis(),
    )
    .unwrap_or(u64::MAX);

    if tiers.dp_device_blocks.as_slice() != [(worker.dp_rank, tiers.gpu_blocks)].as_slice() {
        return Err(SpeculativeOnboardingRejectReason::UnsupportedDpTopology);
    }

    let block_size = usize::try_from(block_size)
        .ok()
        .filter(|value| *value > 0)
        .ok_or(SpeculativeOnboardingRejectReason::BlockHashBounds)?;
    let max_reusable_blocks = isl_tokens.saturating_sub(1) / block_size;
    let host_blocks = usize::try_from(tiers.host_pinned_blocks)
        .unwrap_or(usize::MAX)
        .min(max_reusable_blocks);
    let disk_blocks = usize::try_from(tiers.disk_blocks)
        .unwrap_or(usize::MAX)
        .min(max_reusable_blocks);
    if disk_blocks <= host_blocks {
        return Err(SpeculativeOnboardingRejectReason::NoDiskBlocks);
    }

    let sequence_hashes =
        sequence_hashes.ok_or(SpeculativeOnboardingRejectReason::MissingBlockHashes)?;
    let disk_hashes = sequence_hashes
        .get(host_blocks..disk_blocks)
        .ok_or(SpeculativeOnboardingRejectReason::BlockHashBounds)?
        .to_vec();
    let predicted_disk_blocks = u32::try_from(disk_hashes.len())
        .map_err(|_| SpeculativeOnboardingRejectReason::BlockHashBounds)?;
    if predicted_disk_blocks == 0 {
        return Err(SpeculativeOnboardingRejectReason::NoDiskBlocks);
    }

    Ok(SpeculativeOnboardingNotice {
        schema: M2_NOTICE_SCHEMA,
        request_id: request_id.to_string(),
        worker_id: worker.worker_id,
        dp_rank: worker.dp_rank,
        identity: identity.into(),
        worker_epoch,
        state_version,
        state_age_ms,
        prefix_start_block: u32::try_from(host_blocks)
            .map_err(|_| SpeculativeOnboardingRejectReason::BlockHashBounds)?,
        block_hashes: disk_hashes,
        predicted_disk_blocks,
        predicted_queue_window_ms: None,
        estimated_stage_ms: None,
        deadline_budget_ms: None,
        policy: SpeculativeOnboardingPolicy::DryRun,
    })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::scheduling::LowerTierFallbackReason;

    fn known_state() -> LowerTierStateView {
        LowerTierStateView {
            status: LowerTierStateStatus::Known,
            version: Some(7),
            age: Some(Duration::from_millis(12)),
            worker_epoch: Some(3),
            fallback_reason: LowerTierFallbackReason::None,
        }
    }

    fn tiers(gpu: u32, host: u32, disk: u32) -> SelectedWorkerTierSnapshot {
        SelectedWorkerTierSnapshot {
            dp_device_blocks: vec![(0, gpu)],
            gpu_blocks: gpu,
            host_pinned_blocks: host,
            disk_blocks: disk,
        }
    }

    fn qualify(
        state: &LowerTierStateView,
        tiers: &SelectedWorkerTierSnapshot,
        hashes: Option<&[SequenceHash]>,
        isl_tokens: usize,
    ) -> Result<SpeculativeOnboardingNotice, SpeculativeOnboardingRejectReason> {
        qualify_speculative_onboarding_notice(
            Some("request-1"),
            WorkerWithDpRank::new(9, 0),
            &CacheIdentity::new("model", "", None::<String>),
            state,
            tiers,
            hashes,
            isl_tokens,
            4,
        )
    }

    #[test]
    fn known_disk_uses_only_cumulative_disk_suffix() {
        let notice = qualify(
            &known_state(),
            &tiers(1, 2, 5),
            Some(&[10, 20, 30, 40, 50]),
            25,
        )
        .unwrap();

        assert_eq!(notice.prefix_start_block, 2);
        assert_eq!(notice.block_hashes, vec![30, 40, 50]);
        assert_eq!(notice.predicted_disk_blocks, 3);
    }

    #[test]
    fn block_aligned_prompt_drops_last_disk_block() {
        let notice = qualify(&known_state(), &tiers(0, 1, 3), Some(&[10, 20, 30]), 12).unwrap();

        assert_eq!(notice.block_hashes, vec![20]);
        assert_eq!(notice.predicted_disk_blocks, 1);
    }

    #[test]
    fn non_known_states_fail_closed() {
        for (status, expected) in [
            (
                LowerTierStateStatus::Unknown,
                SpeculativeOnboardingRejectReason::StateUnknown,
            ),
            (
                LowerTierStateStatus::Stale,
                SpeculativeOnboardingRejectReason::StateStale,
            ),
            (
                LowerTierStateStatus::Conflict,
                SpeculativeOnboardingRejectReason::StateConflict,
            ),
        ] {
            let mut state = known_state();
            state.status = status;
            assert_eq!(
                qualify(&state, &tiers(0, 0, 1), Some(&[10]), 8),
                Err(expected)
            );
        }
    }

    #[test]
    fn identity_mismatch_has_specific_reject_reason() {
        let mut state = known_state();
        state.status = LowerTierStateStatus::Unknown;
        state.fallback_reason = LowerTierFallbackReason::IdentityMismatch;
        assert_eq!(
            qualify(&state, &tiers(0, 0, 1), Some(&[10]), 8),
            Err(SpeculativeOnboardingRejectReason::IdentityMismatch)
        );
    }

    #[test]
    fn host_only_and_missing_or_short_hashes_reject() {
        assert_eq!(
            qualify(&known_state(), &tiers(0, 2, 2), Some(&[10, 20]), 12),
            Err(SpeculativeOnboardingRejectReason::NoDiskBlocks)
        );
        assert_eq!(
            qualify(&known_state(), &tiers(0, 0, 1), None, 8),
            Err(SpeculativeOnboardingRejectReason::MissingBlockHashes)
        );
        assert_eq!(
            qualify(&known_state(), &tiers(0, 0, 2), Some(&[10]), 12),
            Err(SpeculativeOnboardingRejectReason::BlockHashBounds)
        );
    }

    #[test]
    fn multi_dp_snapshot_rejects() {
        let mut multi = tiers(1, 2, 3);
        multi.dp_device_blocks.push((1, 1));
        assert_eq!(
            qualify(&known_state(), &multi, Some(&[10, 20, 30]), 16),
            Err(SpeculativeOnboardingRejectReason::UnsupportedDpTopology)
        );
    }

    #[test]
    fn notice_schema_round_trips_and_rejects_unknown_fields() {
        let notice = qualify(&known_state(), &tiers(0, 0, 1), Some(&[10]), 8).unwrap();
        let value = serde_json::to_value(&notice).unwrap();
        assert_eq!(
            serde_json::from_value::<SpeculativeOnboardingNotice>(value.clone()).unwrap(),
            notice
        );

        let mut value = value.as_object().unwrap().clone();
        value.insert("unexpected".to_string(), serde_json::json!(true));
        assert!(serde_json::from_value::<SpeculativeOnboardingNotice>(value.into()).is_err());
    }
}
