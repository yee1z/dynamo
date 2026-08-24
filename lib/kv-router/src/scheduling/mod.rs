// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod config;
mod filter;
mod local;
pub mod lower_tier_state;
pub mod overlap;
pub mod overlap_refresh;
pub mod policy;
pub mod policy_config;
pub mod policy_queue;
pub mod prefill_load;
pub mod queue;
pub mod selector;
pub mod speculative_onboarding;

mod types;
pub use filter::*;
pub use local::LocalScheduler;
pub use lower_tier_state::{
    CacheIdentity, LowerTierApplyResult, LowerTierFallbackReason, LowerTierPredictionDisposition,
    LowerTierReadTicket, LowerTierStateLedger, LowerTierStateStatus, LowerTierStateUpdate,
    LowerTierStateView, LowerTierUpdateFence, LowerTierUpdateKind, qualify_lower_tier_prediction,
};
pub use overlap::{
    CacheHitEstimates, OverlapAnalysis, OverlapScoresResponse, OverlapSignals,
    SelectedWorkerTierSnapshot, SharedCacheOverlapScore, WorkerOverlapScore,
};
pub use overlap_refresh::{
    NoopOverlapScoresRefresh, OverlapScoresRefresh, RefreshedOverlap, TieredOverlapRefresher,
};
pub use policy_config::{
    PolicyClassConfig, PolicyProfile, RouterPolicyConfig, RouterPolicyConfigError,
};
pub use policy_queue::{
    PolicyQueue, PolicyQueueEntry, QueueLimitKind, QueueRejection, QueueSnapshot,
};
pub use prefill_load::{
    InvalidEffectivePrefillTokens, PrefillLoadEstimator, effective_prefill_tokens,
    prefill_load_hint_from_effective_tokens,
};
pub use speculative_onboarding::{
    DYN_M2_DRY_RUN_NOTICE, DYN_M2_POLICY, M2_NOTICE_EXTRA_ARGS_KEY, M2_NOTICE_SCHEMA,
    SpeculativeCacheIdentity, SpeculativeOnboardingNotice, SpeculativeOnboardingPolicy,
    SpeculativeOnboardingRejectReason, configured_speculative_onboarding_policy,
    qualify_speculative_onboarding_notice, speculative_onboarding_dry_run_enabled,
};
pub use types::*;
