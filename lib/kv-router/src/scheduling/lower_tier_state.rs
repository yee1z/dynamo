// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conservative lower-tier state contract for prediction-only routing.
//!
//! This module deliberately carries metadata only. It neither owns KV blocks nor
//! starts transfers. A caller may use a [`LowerTierStateView`] to qualify an
//! indexer match, but every non-`known` state must remain on the passive path.

use std::sync::RwLock;
use std::time::{Duration, Instant};

use rustc_hash::FxHashMap;

use crate::protocols::WorkerWithDpRank;

/// Cache identity fields that must agree before lower-tier state is reusable.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheIdentity {
    pub model: String,
    pub salt: String,
    pub adapter: Option<String>,
}

impl CacheIdentity {
    pub fn new(
        model: impl Into<String>,
        salt: impl Into<String>,
        adapter: Option<impl Into<String>>,
    ) -> Self {
        Self {
            model: model.into(),
            salt: salt.into(),
            adapter: adapter.map(Into::into),
        }
    }

    pub fn with_adapter(&self, adapter: Option<String>) -> Self {
        Self {
            model: self.model.clone(),
            salt: self.salt.clone(),
            adapter,
        }
    }
}

/// How an update relates to the worker's state stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowerTierUpdateKind {
    /// Authoritative full state for an epoch.
    Snapshot,
    /// Insert, offload, onboard, promotion, eviction, or removal delta.
    Incremental,
    /// Worker restart or cache reset barrier. A new snapshot is required.
    Reset,
}

impl LowerTierUpdateKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Snapshot => "snapshot",
            Self::Incremental => "incremental",
            Self::Reset => "reset",
        }
    }
}

/// Metadata-only update emitted by a worker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LowerTierStateUpdate {
    pub worker: WorkerWithDpRank,
    pub identity: CacheIdentity,
    pub kind: LowerTierUpdateKind,
    pub worker_epoch: u64,
    pub version: u64,
    /// Worker-local monotonic timestamp retained for diagnostics only.
    pub updated_ns: u64,
    /// Stable digest of the worker's conservative prefix/block summary.
    pub summary_digest: u64,
}

/// Router-side status of a worker's lower-tier metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowerTierStateStatus {
    Known,
    Unknown,
    Stale,
    Conflict,
}

impl LowerTierStateStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Known => "known",
            Self::Unknown => "unknown",
            Self::Stale => "stale",
            Self::Conflict => "conflict",
        }
    }
}

/// Stable fallback reason recorded in the request trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowerTierFallbackReason {
    None,
    AwaitingInitialSnapshot,
    AwaitingSnapshotAfterReset,
    PropagationFailure,
    Expired,
    IdentityMismatch,
    ConflictingReplay,
}

impl LowerTierFallbackReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::AwaitingInitialSnapshot => "awaiting_initial_snapshot",
            Self::AwaitingSnapshotAfterReset => "awaiting_snapshot_after_reset",
            Self::PropagationFailure => "propagation_failure",
            Self::Expired => "expired",
            Self::IdentityMismatch => "identity_mismatch",
            Self::ConflictingReplay => "conflicting_replay",
        }
    }
}

/// Immutable request-time view used by scheduling and tracing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LowerTierStateView {
    pub status: LowerTierStateStatus,
    pub version: Option<u64>,
    pub age: Option<Duration>,
    pub worker_epoch: Option<u64>,
    pub fallback_reason: LowerTierFallbackReason,
}

impl LowerTierStateView {
    pub fn unknown(reason: LowerTierFallbackReason) -> Self {
        Self {
            status: LowerTierStateStatus::Unknown,
            version: None,
            age: None,
            worker_epoch: None,
            fallback_reason: reason,
        }
    }

    pub fn confidence(&self) -> f64 {
        if self.status == LowerTierStateStatus::Known {
            1.0
        } else {
            0.0
        }
    }
}

impl Default for LowerTierStateView {
    fn default() -> Self {
        Self::unknown(LowerTierFallbackReason::AwaitingInitialSnapshot)
    }
}

/// Result of applying an update to the router ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowerTierApplyResult {
    AppliedKnown,
    AppliedUnknown,
    Duplicate,
    RejectedOldEpoch,
    RejectedOutOfOrder,
    Conflict,
}

impl LowerTierApplyResult {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AppliedKnown => "applied_known",
            Self::AppliedUnknown => "applied_unknown",
            Self::Duplicate => "duplicate",
            Self::RejectedOldEpoch => "rejected_old_epoch",
            Self::RejectedOutOfOrder => "rejected_out_of_order",
            Self::Conflict => "conflict",
        }
    }
}

#[derive(Debug, Clone)]
struct LedgerEntry {
    identity: CacheIdentity,
    kind: LowerTierUpdateKind,
    worker_epoch: u64,
    version: u64,
    updated_ns: u64,
    summary_digest: u64,
    observed_at: Instant,
    status: LowerTierStateStatus,
    fallback_reason: LowerTierFallbackReason,
    transitioning_from_known: bool,
}

impl LedgerEntry {
    fn matches(&self, update: &LowerTierStateUpdate) -> bool {
        self.identity == update.identity
            && self.kind == update.kind
            && self.worker_epoch == update.worker_epoch
            && self.version == update.version
            && self.updated_ns == update.updated_ns
            && self.summary_digest == update.summary_digest
    }

    fn from_update(
        update: &LowerTierStateUpdate,
        observed_at: Instant,
        status: LowerTierStateStatus,
        fallback_reason: LowerTierFallbackReason,
    ) -> Self {
        Self {
            identity: update.identity.clone(),
            kind: update.kind,
            worker_epoch: update.worker_epoch,
            version: update.version,
            updated_ns: update.updated_ns,
            summary_digest: update.summary_digest,
            observed_at,
            status,
            fallback_reason,
            transitioning_from_known: false,
        }
    }
}

/// Thread-safe worker state ledger with a fail-closed freshness policy.
pub struct LowerTierStateLedger {
    entries: RwLock<FxHashMap<WorkerWithDpRank, LedgerEntry>>,
    stale_after: Duration,
}

impl LowerTierStateLedger {
    pub fn new(stale_after: Duration) -> Self {
        Self {
            entries: RwLock::new(FxHashMap::default()),
            stale_after,
        }
    }

    /// Apply a snapshot, delta, or reset without ever moving cache data.
    pub fn apply(
        &self,
        update: LowerTierStateUpdate,
        observed_at: Instant,
    ) -> LowerTierApplyResult {
        let mut entries = self.entries.write().unwrap();
        let existing = entries.get(&update.worker);

        if let Some(existing) = existing {
            if update.worker_epoch < existing.worker_epoch {
                return LowerTierApplyResult::RejectedOldEpoch;
            }
            if update.worker_epoch == existing.worker_epoch {
                if update.version < existing.version {
                    return LowerTierApplyResult::RejectedOutOfOrder;
                }
                if update.version == existing.version {
                    if existing.matches(&update) {
                        return LowerTierApplyResult::Duplicate;
                    }
                    let mut conflict = existing.clone();
                    conflict.status = LowerTierStateStatus::Conflict;
                    conflict.fallback_reason = LowerTierFallbackReason::ConflictingReplay;
                    conflict.transitioning_from_known = false;
                    entries.insert(update.worker, conflict);
                    return LowerTierApplyResult::Conflict;
                }
                if update.identity != existing.identity {
                    let conflict = LedgerEntry::from_update(
                        &update,
                        observed_at,
                        LowerTierStateStatus::Conflict,
                        LowerTierFallbackReason::ConflictingReplay,
                    );
                    entries.insert(update.worker, conflict);
                    return LowerTierApplyResult::Conflict;
                }
            }
        }

        let epoch_advanced = existing.is_some_and(|entry| update.worker_epoch > entry.worker_epoch);
        let (status, fallback_reason, result) = match update.kind {
            LowerTierUpdateKind::Snapshot => (
                LowerTierStateStatus::Known,
                LowerTierFallbackReason::None,
                LowerTierApplyResult::AppliedKnown,
            ),
            LowerTierUpdateKind::Reset => (
                LowerTierStateStatus::Unknown,
                LowerTierFallbackReason::AwaitingSnapshotAfterReset,
                LowerTierApplyResult::AppliedUnknown,
            ),
            LowerTierUpdateKind::Incremental
                if existing.is_none()
                    || epoch_advanced
                    || existing.is_some_and(|entry| {
                        entry.status != LowerTierStateStatus::Known
                            && !entry.transitioning_from_known
                    }) =>
            {
                (
                    LowerTierStateStatus::Unknown,
                    LowerTierFallbackReason::AwaitingSnapshotAfterReset,
                    LowerTierApplyResult::AppliedUnknown,
                )
            }
            LowerTierUpdateKind::Incremental => (
                LowerTierStateStatus::Known,
                LowerTierFallbackReason::None,
                LowerTierApplyResult::AppliedKnown,
            ),
        };

        entries.insert(
            update.worker,
            LedgerEntry::from_update(&update, observed_at, status, fallback_reason),
        );
        result
    }

    /// Fail closed after a propagation error while preserving the last watermark.
    pub fn mark_propagation_failure(&self, worker: WorkerWithDpRank) {
        if let Some(entry) = self.entries.write().unwrap().get_mut(&worker) {
            entry.status = LowerTierStateStatus::Unknown;
            entry.fallback_reason = LowerTierFallbackReason::PropagationFailure;
            entry.transitioning_from_known = false;
        }
    }

    /// Temporarily fail closed while the indexer applies an accepted event.
    /// A same-epoch incremental may restore `known`; a real propagation
    /// failure cannot.
    pub fn begin_update(&self, worker: WorkerWithDpRank) {
        if let Some(entry) = self.entries.write().unwrap().get_mut(&worker)
            && entry.status == LowerTierStateStatus::Known
        {
            entry.status = LowerTierStateStatus::Unknown;
            entry.fallback_reason = LowerTierFallbackReason::PropagationFailure;
            entry.transitioning_from_known = true;
        }
    }

    /// Query a worker using router receipt time for freshness. Worker clocks are
    /// diagnostic only and are never compared across processes.
    pub fn view(
        &self,
        worker: WorkerWithDpRank,
        expected_identity: &CacheIdentity,
        now: Instant,
    ) -> LowerTierStateView {
        let entries = self.entries.read().unwrap();
        let Some(entry) = entries.get(&worker) else {
            return LowerTierStateView::default();
        };
        let age = now.saturating_duration_since(entry.observed_at);
        let base = LowerTierStateView {
            status: entry.status,
            version: Some(entry.version),
            age: Some(age),
            worker_epoch: Some(entry.worker_epoch),
            fallback_reason: entry.fallback_reason,
        };
        if entry.status == LowerTierStateStatus::Conflict {
            return base;
        }
        if &entry.identity != expected_identity {
            return LowerTierStateView {
                status: LowerTierStateStatus::Conflict,
                fallback_reason: LowerTierFallbackReason::IdentityMismatch,
                ..base
            };
        }
        if entry.status != LowerTierStateStatus::Known {
            return base;
        }
        if age > self.stale_after {
            return LowerTierStateView {
                status: LowerTierStateStatus::Stale,
                fallback_reason: LowerTierFallbackReason::Expired,
                ..base
            };
        }
        base
    }

    /// Snapshot every worker view at one router-local instant. Used when a
    /// queued request refreshes its overlap scores before dispatch.
    pub fn views(
        &self,
        expected_identity: &CacheIdentity,
        now: Instant,
    ) -> FxHashMap<WorkerWithDpRank, LowerTierStateView> {
        let workers = self
            .entries
            .read()
            .unwrap()
            .keys()
            .copied()
            .collect::<Vec<_>>();
        workers
            .into_iter()
            .map(|worker| (worker, self.view(worker, expected_identity, now)))
            .collect()
    }
}

/// Whether an observed match can be interpreted as a verified prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowerTierPredictionDisposition {
    ConfirmedHit,
    VerifiedMiss,
    PassiveFallback,
}

impl LowerTierPredictionDisposition {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ConfirmedHit => "confirmed_hit",
            Self::VerifiedMiss => "verified_miss",
            Self::PassiveFallback => "passive_fallback",
        }
    }
}

pub fn qualify_lower_tier_prediction(
    state: &LowerTierStateView,
    matched_blocks: usize,
) -> LowerTierPredictionDisposition {
    if state.status != LowerTierStateStatus::Known {
        return LowerTierPredictionDisposition::PassiveFallback;
    }
    if matched_blocks == 0 {
        LowerTierPredictionDisposition::VerifiedMiss
    } else {
        LowerTierPredictionDisposition::ConfirmedHit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    fn worker() -> WorkerWithDpRank {
        WorkerWithDpRank::new(7, 1)
    }

    fn identity(adapter: Option<&str>) -> CacheIdentity {
        CacheIdentity::new("Qwen/Qwen3-8B", "salt-v1", adapter)
    }

    fn update(
        kind: LowerTierUpdateKind,
        epoch: u64,
        version: u64,
        digest: u64,
    ) -> LowerTierStateUpdate {
        LowerTierStateUpdate {
            worker: worker(),
            identity: identity(None),
            kind,
            worker_epoch: epoch,
            version,
            updated_ns: version * 10,
            summary_digest: digest,
        }
    }

    #[test]
    fn initial_snapshot_distinguishes_unknown_miss_and_hit() {
        let ledger = LowerTierStateLedger::new(Duration::from_secs(10));
        let now = Instant::now();
        let unknown = ledger.view(worker(), &identity(None), now);
        assert_eq!(unknown.status, LowerTierStateStatus::Unknown);
        assert_eq!(
            qualify_lower_tier_prediction(&unknown, 0),
            LowerTierPredictionDisposition::PassiveFallback
        );

        assert_eq!(
            ledger.apply(update(LowerTierUpdateKind::Snapshot, 1, 1, 101), now),
            LowerTierApplyResult::AppliedKnown
        );
        let known = ledger.view(worker(), &identity(None), now);
        assert_eq!(known.status, LowerTierStateStatus::Known);
        assert_eq!(
            qualify_lower_tier_prediction(&known, 0),
            LowerTierPredictionDisposition::VerifiedMiss
        );
        assert_eq!(
            qualify_lower_tier_prediction(&known, 12),
            LowerTierPredictionDisposition::ConfirmedHit
        );
    }

    #[test]
    fn duplicate_and_out_of_order_updates_do_not_regress_state() {
        let ledger = LowerTierStateLedger::new(Duration::from_secs(10));
        let now = Instant::now();
        let snapshot = update(LowerTierUpdateKind::Snapshot, 4, 10, 100);
        assert_eq!(
            ledger.apply(snapshot.clone(), now),
            LowerTierApplyResult::AppliedKnown
        );
        assert_eq!(
            ledger.apply(snapshot, now + Duration::from_millis(1)),
            LowerTierApplyResult::Duplicate
        );
        assert_eq!(
            ledger.apply(
                update(LowerTierUpdateKind::Incremental, 4, 9, 99),
                now + Duration::from_millis(2)
            ),
            LowerTierApplyResult::RejectedOutOfOrder
        );
        let view = ledger.view(worker(), &identity(None), now + Duration::from_millis(3));
        assert_eq!(view.version, Some(10));
        assert_eq!(view.status, LowerTierStateStatus::Known);
    }

    #[test]
    fn conflicting_same_version_replay_fails_closed() {
        let ledger = LowerTierStateLedger::new(Duration::from_secs(10));
        let now = Instant::now();
        ledger.apply(update(LowerTierUpdateKind::Snapshot, 1, 2, 20), now);
        assert_eq!(
            ledger.apply(
                update(LowerTierUpdateKind::Snapshot, 1, 2, 21),
                now + Duration::from_millis(1)
            ),
            LowerTierApplyResult::Conflict
        );
        let view = ledger.view(worker(), &identity(None), now + Duration::from_millis(2));
        assert_eq!(view.status, LowerTierStateStatus::Conflict);
        assert_eq!(
            view.fallback_reason,
            LowerTierFallbackReason::ConflictingReplay
        );
    }

    #[test]
    fn restart_and_reset_require_a_new_snapshot() {
        let ledger = LowerTierStateLedger::new(Duration::from_secs(10));
        let now = Instant::now();
        ledger.apply(update(LowerTierUpdateKind::Snapshot, 1, 8, 80), now);
        assert_eq!(
            ledger.apply(
                update(LowerTierUpdateKind::Incremental, 2, 1, 90),
                now + Duration::from_millis(1)
            ),
            LowerTierApplyResult::AppliedUnknown
        );
        assert_eq!(
            ledger.apply(
                update(LowerTierUpdateKind::Snapshot, 1, 9, 99),
                now + Duration::from_millis(2)
            ),
            LowerTierApplyResult::RejectedOldEpoch
        );
        assert_eq!(
            ledger
                .view(worker(), &identity(None), now + Duration::from_millis(3))
                .status,
            LowerTierStateStatus::Unknown
        );
        ledger.apply(
            update(LowerTierUpdateKind::Snapshot, 2, 2, 100),
            now + Duration::from_millis(4),
        );
        assert_eq!(
            ledger
                .view(worker(), &identity(None), now + Duration::from_millis(5))
                .status,
            LowerTierStateStatus::Known
        );
        ledger.apply(
            update(LowerTierUpdateKind::Reset, 2, 3, 0),
            now + Duration::from_millis(6),
        );
        assert_eq!(
            ledger
                .view(worker(), &identity(None), now + Duration::from_millis(7))
                .status,
            LowerTierStateStatus::Unknown
        );
    }

    #[test]
    fn promotion_and_eviction_watermarks_advance_without_changing_epoch() {
        let ledger = LowerTierStateLedger::new(Duration::from_secs(10));
        let now = Instant::now();
        ledger.apply(update(LowerTierUpdateKind::Snapshot, 3, 1, 100), now);
        assert_eq!(
            ledger.apply(
                update(LowerTierUpdateKind::Incremental, 3, 2, 200),
                now + Duration::from_millis(1)
            ),
            LowerTierApplyResult::AppliedKnown
        );
        assert_eq!(
            ledger.apply(
                update(LowerTierUpdateKind::Incremental, 3, 3, 300),
                now + Duration::from_millis(2)
            ),
            LowerTierApplyResult::AppliedKnown
        );
        let view = ledger.view(worker(), &identity(None), now + Duration::from_millis(3));
        assert_eq!(view.version, Some(3));
        assert_eq!(view.worker_epoch, Some(3));
        assert_eq!(view.status, LowerTierStateStatus::Known);
    }

    #[test]
    fn stale_identity_mismatch_and_propagation_failure_are_passive() {
        let ledger = LowerTierStateLedger::new(Duration::from_millis(10));
        let now = Instant::now();
        ledger.apply(update(LowerTierUpdateKind::Snapshot, 1, 1, 10), now);

        let mismatched = ledger.view(worker(), &identity(Some("adapter-a")), now);
        assert_eq!(mismatched.status, LowerTierStateStatus::Conflict);
        assert_eq!(
            mismatched.fallback_reason,
            LowerTierFallbackReason::IdentityMismatch
        );

        let stale = ledger.view(worker(), &identity(None), now + Duration::from_millis(11));
        assert_eq!(stale.status, LowerTierStateStatus::Stale);
        assert_eq!(stale.fallback_reason, LowerTierFallbackReason::Expired);

        ledger.mark_propagation_failure(worker());
        let failed = ledger.view(worker(), &identity(None), now + Duration::from_millis(1));
        assert_eq!(failed.status, LowerTierStateStatus::Unknown);
        assert_eq!(
            qualify_lower_tier_prediction(&failed, 8),
            LowerTierPredictionDisposition::PassiveFallback
        );
    }

    #[test]
    fn concurrent_reads_and_updates_return_complete_views() {
        let ledger = Arc::new(LowerTierStateLedger::new(Duration::from_secs(10)));
        let now = Instant::now();
        ledger.apply(update(LowerTierUpdateKind::Snapshot, 1, 1, 10), now);

        let writer = {
            let ledger = ledger.clone();
            thread::spawn(move || {
                for version in 2..=1_000 {
                    ledger.apply(
                        update(LowerTierUpdateKind::Incremental, 1, version, version),
                        Instant::now(),
                    );
                }
            })
        };
        let reader = {
            let ledger = ledger.clone();
            thread::spawn(move || {
                for _ in 0..1_000 {
                    let view = ledger.view(worker(), &identity(None), Instant::now());
                    assert_eq!(view.status, LowerTierStateStatus::Known);
                    assert!(view.version.is_some());
                    assert_eq!(view.worker_epoch, Some(1));
                }
            })
        };

        writer.join().unwrap();
        reader.join().unwrap();
        assert_eq!(
            ledger
                .view(worker(), &identity(None), Instant::now())
                .version,
            Some(1_000)
        );
    }

    #[test]
    fn in_flight_index_update_is_passive_then_incremental_restores_known() {
        let ledger = LowerTierStateLedger::new(Duration::from_secs(10));
        let now = Instant::now();
        ledger.apply(update(LowerTierUpdateKind::Snapshot, 1, 1, 10), now);

        ledger.begin_update(worker());
        let in_flight = ledger.view(worker(), &identity(None), now);
        assert_eq!(in_flight.status, LowerTierStateStatus::Unknown);
        assert_eq!(
            qualify_lower_tier_prediction(&in_flight, 4),
            LowerTierPredictionDisposition::PassiveFallback
        );

        assert_eq!(
            ledger.apply(
                update(LowerTierUpdateKind::Incremental, 1, 2, 20),
                now + Duration::from_millis(1),
            ),
            LowerTierApplyResult::AppliedKnown
        );
        assert_eq!(
            ledger
                .view(worker(), &identity(None), now + Duration::from_millis(2))
                .status,
            LowerTierStateStatus::Known
        );
    }
}
