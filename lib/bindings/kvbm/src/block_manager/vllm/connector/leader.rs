// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod recorder;
pub mod slot;

use super::*;
use dynamo_llm::block_manager::metrics_kvbm::{KvbmMetrics, KvbmMetricsRegistry};
use dynamo_llm::kv_router::scheduling::{
    M2_NOTICE_SCHEMA, SpeculativeOnboardingNotice, SpeculativeOnboardingPolicy,
};
use slot::{ConnectorSlotManager, SlotError, SlotManager, SlotState};

use crate::block_manager::BlockManagerBuilder;
use crate::block_manager::{
    VllmBlockManager, distributed::KvbmLeader as PyKvbmLeader, vllm::KvbmRequest,
    vllm::connector::leader::slot::VllmConnectorSlot,
};
use crate::get_current_tokio_handle;

use dynamo_llm::block_manager::{
    BasicMetadata, DiskStorage, ImmutableBlock, PinnedStorage,
    block::{
        data::logical::distributed_leader_worker::DistributedLeaderWorkerResources,
        locality::Logical,
    },
    connector::{protocol::RequestType, *},
    kv_consolidator::{EventSource, KvEventConsolidationMode},
};
use dynamo_llm::tokens::{SaltHash, TokenBlockSequence, Tokens};
use dynamo_runtime::config::environment_names::kvbm as env_kvbm;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use std::{
    collections::{HashMap, HashSet},
    sync::Mutex,
};
use tokio;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;
use tokio::runtime::Handle;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

type VllmLocality = Logical<DistributedLeaderWorkerResources>;

fn parse_consolidator_mode(mode: Option<String>) -> KvEventConsolidationMode {
    let Some(mode) = mode else {
        return KvEventConsolidationMode::Dedup;
    };

    match mode.parse() {
        Ok(mode) => mode,
        Err(error) => {
            tracing::warn!(
                "Invalid KV event consolidator mode {:?}: {}. Falling back to dedup.",
                mode,
                error
            );
            KvEventConsolidationMode::Dedup
        }
    }
}

impl From<SlotError> for PyErr {
    fn from(err: SlotError) -> Self {
        to_pyerr(err)
    }
}
use anyhow;
use dynamo_llm::recorder::Recorder;
use tokio_util::sync::CancellationToken;

pub trait Leader: Send + Sync + std::fmt::Debug {
    fn get_num_new_matched_tokens(
        &self,
        request_id: String,
        request_num_tokens: usize,
        num_computed_tokens: usize,
    ) -> anyhow::Result<(usize, bool)>;

    fn update_state_after_alloc(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> anyhow::Result<()>;

    fn build_connector_metadata(
        &mut self,
        scheduler_output: SchedulerOutput,
    ) -> anyhow::Result<Vec<u8>>;

    fn request_finished(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
    ) -> anyhow::Result<bool>;

    fn has_slot(&self, request_id: String) -> bool;

    fn create_slot(&mut self, request: KvbmRequest, tokens: Vec<u32>) -> anyhow::Result<()>;

    /// Reset KVBM-managed prefix cache state.
    ///
    /// Returns `Ok(false)` when KVBM rejects the reset so vLLM can propagate
    /// connector reset failure through its `reset_cache` boolean contract.
    fn reset_cache(&mut self) -> anyhow::Result<bool>;

    fn slot_manager(&self) -> &ConnectorSlotManager<String>;

    fn m2_staging_manager(&self) -> Option<Arc<M2StagingManager>> {
        None
    }
}

#[derive(Debug)]
pub struct KvConnectorLeader {
    slot_manager: Arc<OnceLock<ConnectorSlotManager<String>>>,
    block_size: usize,
    inflight_requests: HashSet<String>,
    onboarding_slots: HashSet<String>,
    iteration_counter: u64,
    kvbm_metrics: KvbmMetrics,
    m2_staging: Arc<OnceLock<Arc<M2StagingManager>>>,
}

impl KvConnectorLeader {
    fn new(
        worker_id: String,
        page_size: usize,
        leader_py: PyKvbmLeader,
        consolidator_vllm_endpoint: Option<String>,
        consolidator_output_endpoint: Option<String>,
        consolidator_mode: Option<String>,
    ) -> Self {
        tracing::info!(
            "KvConnectorLeader initialized with worker_id: {}",
            worker_id
        );

        let leader = leader_py.get_inner().clone();
        let handle: Handle = get_current_tokio_handle();

        let kvbm_metrics = KvbmMetrics::new(
            &KvbmMetricsRegistry::default(),
            kvbm_metrics_endpoint_enabled(),
            parse_kvbm_metrics_port(),
        );
        let kvbm_metrics_clone = kvbm_metrics.clone();

        let slot_manager_cell = Arc::new(OnceLock::new());
        let m2_staging_cell = Arc::new(OnceLock::new());
        let (leader_ready_tx, leader_ready_rx) = oneshot::channel::<String>();

        {
            let slot_manager_cell = slot_manager_cell.clone();
            let m2_staging_cell = m2_staging_cell.clone();
            // Capture consolidator endpoints for the async block
            let consolidator_vllm_ep = consolidator_vllm_endpoint.clone();
            let consolidator_output_ep = consolidator_output_endpoint.clone();
            let consolidator_mode = parse_consolidator_mode(consolidator_mode.clone());
            let staging_handle = handle.clone();

            handle.spawn(async move {
                let ready = leader.wait_worker_sync_ready().await;
                if !ready {
                    tracing::error!(
                        "KvConnectorLeader init aborted: leader worker barrier not ready!",
                    );
                    return;
                }

                let mut block_manager_builder = BlockManagerBuilder::new()
                    .worker_id(0)
                    .leader(leader_py)
                    .page_size(page_size)
                    .disable_device_pool(false)
                    .kvbm_metrics(kvbm_metrics_clone.clone());

                // Add consolidator config if provided
                if let (Some(vllm_ep), Some(output_ep)) =
                    (consolidator_vllm_ep, consolidator_output_ep)
                {
                    tracing::debug!(
                        "Adding consolidator config to BlockManager: vllm={}, output={}",
                        vllm_ep,
                        output_ep
                    );
                    block_manager_builder = block_manager_builder.consolidator_config(
                        vllm_ep,
                        Some(output_ep),
                        EventSource::Vllm,
                        consolidator_mode,
                    );
                }

                let block_manager = match block_manager_builder.build().await {
                    Ok(bm) => bm,
                    Err(e) => {
                        tracing::error!("Failed to build BlockManager: {}", e);
                        return;
                    }
                };

                // Create the slot manager now that everything is ready
                let sm = ConnectorSlotManager::new(
                    block_manager.get_block_manager().clone(),
                    leader.clone(),
                    kvbm_metrics_clone.clone(),
                    Some(format!("worker-{}", worker_id)), // identifier for cache stats
                );

                if m2_side_effect_enabled() {
                    let _ = m2_staging_cell.set(Arc::new(M2StagingManager::new(
                        block_manager.get_block_manager().clone(),
                        staging_handle,
                        m2_side_effect_policy().expect("side-effect policy was checked"),
                    )));
                }

                let _ = slot_manager_cell.set(sm);

                if leader_ready_tx.send("finished".to_string()).is_err() {
                    tracing::error!("main routine receiver dropped before result was sent");
                }
            });
        }

        tokio::task::block_in_place(|| {
            handle.block_on(async {
                match leader_ready_rx.await {
                    Ok(_) => tracing::info!("KvConnectorLeader init complete."),
                    Err(_) => tracing::warn!("KvConnectorLeader init channel dropped"),
                }
            });
        });

        Self {
            slot_manager: slot_manager_cell,
            block_size: page_size,
            inflight_requests: HashSet::new(),
            onboarding_slots: HashSet::new(),
            iteration_counter: 0,
            kvbm_metrics,
            m2_staging: m2_staging_cell,
        }
    }
}

impl Leader for KvConnectorLeader {
    #[inline]
    fn slot_manager(&self) -> &ConnectorSlotManager<String> {
        self.slot_manager
            .get()
            .expect("slot_manager not initialized")
    }

    fn m2_staging_manager(&self) -> Option<Arc<M2StagingManager>> {
        self.m2_staging.get().cloned()
    }

    /// Match the tokens in the request with the available block pools.
    /// Note: the necessary details of the request are captured prior to this call. For vllm,
    /// we make a create slot call prior to this call, so a slot is guaranteed to exist.
    ///
    /// To align with the connector interface, we must ensure that if no blocks are matched, we return (0, false).
    /// In our implementation, if we match any block, we return (num_matched_tokens, true).
    #[tracing::instrument(level = "debug", skip(self, request_num_tokens, num_computed_tokens))]
    fn get_num_new_matched_tokens(
        &self,
        request_id: String,
        request_num_tokens: usize,
        num_computed_tokens: usize,
    ) -> anyhow::Result<(usize, bool)> {
        tracing::debug!(
            "request_num_tokens: {request_num_tokens}; num_computed_tokens: {num_computed_tokens}"
        );

        // the number of device matched tokens should be less than or equal to the number of tokens in the request
        debug_assert!(num_computed_tokens.is_multiple_of(self.block_size));

        let shared_slot = self.slot_manager().get_slot(&request_id)?;
        let mut slot = shared_slot
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

        debug_assert!(
            slot.state() != SlotState::Prefilling && slot.state() != SlotState::Decoding,
            "slot is in the Prefilled state or Decoding; shouldn't happen"
        );

        if slot.state() == SlotState::SkippedPrefill || slot.state() == SlotState::SkippedDecode {
            tracing::debug!(
                "slot is in the SkippedPrefill or SkippedDecode state; will resume from skipped and return early"
            );
            match slot.state() {
                SlotState::SkippedPrefill => {
                    slot.mark_as_prefilling(self.iteration_counter)?;
                    return Ok((0, false));
                }
                SlotState::SkippedDecode => {
                    slot.mark_as_decoding(self.iteration_counter)?;
                    return Ok((0, false));
                }
                _ => unreachable!("slot is not in the SkippedPrefill or SkippedDecode state"),
            }
        }

        // early exit if we cannot match full block
        if (slot.sequence().total_tokens() - num_computed_tokens) < self.block_size {
            return Ok((0, false));
        }

        // find matches for any remaining tokens
        // this will advance the computed position and hold any newly matched blocks in the slot
        slot.acquire_local_matches(num_computed_tokens)?;

        // return the number of external tokens that are ready for onboarding
        // we always return true here as we always asynchronously onboard matched blocks
        if let SlotState::OnboardStaged(num_external_tokens) = slot.state() {
            debug_assert!(
                (num_computed_tokens + num_external_tokens).is_multiple_of(self.block_size)
            );
            tracing::debug!(
                request_id = request_id,
                "scheduling onboarding for {} external tokens",
                num_external_tokens
            );
            self.kvbm_metrics
                .matched_tokens
                .inc_by(num_external_tokens as u64);
            Ok((num_external_tokens, true))
        } else {
            Ok((0, false))
        }
    }

    /// Note: vLLM will not provide any scheduler output data for requests that are onboarding. it is entirely
    /// on the connector's implementation to handle this case.
    #[tracing::instrument(level = "debug", skip_all, fields(request_id))]
    fn update_state_after_alloc(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> anyhow::Result<()> {
        tracing::debug!(
            request_id,
            "num_device_blocks: {}; num_external_tokens: {}",
            block_ids.len(),
            num_external_tokens
        );

        let shared_slot = self.slot_manager().get_slot(&request_id)?;
        let mut slot = shared_slot
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

        // we have not yet advanced the computed position, but now we can, since we have an indication that we have
        // necessary gpu blocks into which we will load the external tokens.

        slot.append_mutable_device_blocks(&block_ids)?;

        // the second call will show num_external_tokens == 0
        // this call is just letting us know the other blocks that are being used for the remainder of the prefill
        if num_external_tokens > 0 {
            let num_computed_tokens = block_ids.len() * self.block_size - num_external_tokens;
            slot.record_cached_device_tokens(num_computed_tokens);
            slot.advance_computed_position(num_computed_tokens)?;

            tracing::debug!(
                request_id = request_id,
                "triggering onboarding for {} external tokens",
                num_external_tokens
            );
            slot.trigger_onboarding(num_external_tokens)?;
            self.onboarding_slots.insert(request_id);
        }

        Ok(())
    }

    #[tracing::instrument(level = "debug", skip_all, fields(iteration = self.iteration_counter + 1))]
    fn build_connector_metadata(
        &mut self,
        scheduler_output: SchedulerOutput,
    ) -> anyhow::Result<Vec<u8>> {
        // the iteration counter is used to track the number of times we have built the connector metadata
        // all connetor operations have the iteration counter at which they were issued.
        // this allows operations to be lazily enqueued to the transfer engine
        // the worker side of the connector will track all operations for completion before the request is
        // allowed to be marked as finished.
        self.iteration_counter += 1;
        let iteration = self.iteration_counter;

        tracing::debug!("Building connector metadata");
        tracing::debug!("SchedulerOutput: {scheduler_output:#?}");

        let mut inflight_requests = self.inflight_requests.clone();
        let mut md = ConnectorMetadata::new(iteration);

        let onboarding_slots = std::mem::take(&mut self.onboarding_slots);

        // Worker-side - we create a request slot for onboarding, then delete it when onboarding is finished, then
        // recreate it again when we start the prefill/decode phase.
        //
        // This is kind of a nice abstraction as it keeps the events simplier; however, we now create the request-slot
        // once for onboarding (this loop), then again for prefill/decode (new_requests loop).
        //
        // TODO(krish): Consider a more deterministic way to count immediate ops.
        // Currently we count by filtering pending_ops at runtime. A higher-level approach
        // (e.g., tracking count when onboard_blocks is called, or deriving from architecture
        // config) might be more robust against potential timing-related issues.
        for request_id in onboarding_slots.iter() {
            let shared_slot = self.slot_manager().get_slot(request_id)?;
            let mut slot = shared_slot
                .lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

            let pending_ops_opt = slot.take_pending_operations();

            if let Some(pending_ops) = pending_ops_opt {
                // Count immediate (onboard) operations for this slot
                let num_immediate = pending_ops
                    .iter()
                    .filter(|op| op.request_type == RequestType::Immediate)
                    .count() as u64;

                // Create slot with expected immediate ops BEFORE adding operations
                md.create_slot(request_id.clone(), num_immediate);
                md.add_operations(pending_ops);
            } else {
                // No operations, create slot with 0 expected immediate ops
                md.create_slot(request_id.clone(), 0);
            }

            assert!(
                inflight_requests.remove(request_id),
                "request_id {request_id} not found in inflight_requests: "
            );
        }

        // vLLM provides us with "new_requests" which are "new" after onboarding, but not before or during.
        // this makes the lifecyle a potentially two-phase lifecycle.
        //
        // todo: update the code and abstraction to account for this two-phase lifecycle.
        for new_req in &scheduler_output.new_requests {
            let request_id = &new_req.request_id;

            let already_created = md.new_slots.iter().any(|s| &s.request_id == request_id);

            // Skip if this slot was already created in the onboarding_slots loop above.
            // This prevents overwriting the slot with expected_immediate_ops=0 when it should have the correct count.
            if already_created {
                assert!(
                    inflight_requests.remove(request_id),
                    "request_id {request_id} not found in inflight_requests: "
                );
                continue;
            }

            assert!(
                inflight_requests.remove(request_id),
                "request_id {request_id} not found in inflight_requests: "
            );

            let shared_slot = self.slot_manager().get_slot(request_id)?;
            let mut slot = shared_slot
                .lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

            slot.record_start_iteration(iteration)?;

            debug_assert!(
                matches!(
                    slot.state(),
                    SlotState::Initialized | SlotState::Onboarding(_)
                ),
                "current slot state: {:?}",
                slot.state()
            );

            let scheduled_tokens = *scheduler_output
                .num_scheduled_tokens
                .get(request_id)
                .unwrap_or(&0);

            slot.apply_scheduler_output(
                &[],
                &[],
                new_req.num_computed_tokens,
                scheduled_tokens,
                None,
                None,
            )?;

            let pending_ops_opt = slot.take_pending_operations();

            if let Some(pending_ops) = pending_ops_opt {
                // Count immediate (onboard) operations for this slot
                let num_immediate = pending_ops
                    .iter()
                    .filter(|op| op.request_type == RequestType::Immediate)
                    .count() as u64;

                // Create slot with expected immediate ops BEFORE adding operations
                md.create_slot(new_req.request_id.clone(), num_immediate);
                md.add_operations(pending_ops);
            } else {
                // No operations, create slot with 0 expected immediate ops
                md.create_slot(new_req.request_id.clone(), 0);
            }
        }

        for cached_req in &scheduler_output.cached_requests {
            let request_id = &cached_req.request_id;

            if cached_req.resumed_from_preemption {
                // we really do not know what to expect here:
                // first let's try to get the slot, it might fail because maybe preemption put us thru
                // a finished cycle -- who knows
                let shared_slot = self.slot_manager().get_slot(request_id);
                match &shared_slot {
                    Ok(_) => {
                        tracing::info!("after preemption, slot is still alive");
                    }
                    Err(_) => {
                        tracing::info!("after preemption, slot is not alive");
                    }
                }

                let shared_slot = shared_slot?;
                let mut slot = shared_slot
                    .lock()
                    .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

                // todo: we probably need to reset the slot state and reload it from `cache_req`; however, we do not
                // know if it will take another pass at `get_num_new_matched_tokens` or `update_state_after_alloc`.
                slot.reset_after_preemption();

                // note, we can not trigger onboarding here -- perhaps we are supposed to or perhaps will get another
                // pass at `get_num_new_matched_tokens` or `update_state_after_alloc`.
            }

            assert!(
                inflight_requests.remove(request_id),
                "request_id {request_id} not found in inflight_requests: "
            );

            let shared_slot = self.slot_manager().get_slot(request_id)?;
            let mut slot = shared_slot
                .lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

            let scheduled_tokens = *scheduler_output
                .num_scheduled_tokens
                .get(request_id)
                .unwrap_or(&0);

            slot.apply_scheduler_output(
                &cached_req.new_token_ids,
                &cached_req.new_block_ids,
                cached_req.num_computed_tokens,
                scheduled_tokens,
                None,
                None,
            )?;

            if let Some(pending_ops) = slot.take_pending_operations() {
                tracing::debug!(
                    "adding {} pending operations for slot {}",
                    pending_ops.len(),
                    request_id
                );
                md.add_operations(pending_ops);
            }
        }

        for unscheduled_req in inflight_requests.iter() {
            let shared_slot = self.slot_manager().get_slot(unscheduled_req)?;
            let mut slot_guard = shared_slot
                .lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

            let slot = slot_guard
                .as_any_mut()
                .downcast_mut::<VllmConnectorSlot>()
                .ok_or_else(|| anyhow::anyhow!("Expected VllmConnectorSlot, got different type"))?;

            slot.mark_as_skipped()?;
        }

        tracing::debug!("metadata: {md:#?}");
        serde_json::to_vec(&md)
            .map_err(|e| anyhow::anyhow!("Failed to serialize connector metadata: {}", e))
    }

    fn request_finished(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
    ) -> anyhow::Result<bool> {
        tracing::debug!("Request finished: {request_id}; block_ids: {block_ids:?}");

        if !self.slot_manager().has_slot(&request_id) {
            tracing::warn!(
                "request_finished called for request_id: {request_id} but slot is not found"
            );
            self.inflight_requests.remove(&request_id);
            return Ok(false);
        }

        // grab the slot
        let shared_slot = self.slot_manager().get_slot(&request_id)?;

        // Acquire lock BEFORE marking as finished
        // This ensures we check state and prevent new operations from being created
        let mut slot = shared_slot
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

        // Mark the slot as finished (sets state to Finishing if there are operations,
        // or Finished if all operations are complete)
        slot.mark_as_finished(self.iteration_counter)?;

        // remove the request from the inflight requests
        self.inflight_requests.remove(&request_id);

        // Return value semantics:
        // - `false`: Tells vLLM all GPU blocks are free and the request can be fully cleaned up.
        //            vLLM will immediately remove the request from its internal hash table.
        // - `true`:  Tells vLLM there are outstanding async operations on GPU blocks.
        //            The worker side of the connector API will later call `finish_requests()`
        //            to notify vLLM when the request is truly complete.
        //
        // TODO(jthomson04): This is a temporary fix to ensure vLLM 0.11.2 compatibility.
        //     IMPORTANT: We must ALWAYS return `true` here, even when the slot is already Finished.
        //
        //      Why? If we return `false`, vLLM removes the request from `self.requests` immediately.
        //      However, our worker connector may still report completion later via `finish_requests()`.
        //      When that happens, vLLM's scheduler.py has an assertion `req_id in self.requests`
        //      that will fail because the request was already removed from the hash table.
        //
        //      By always returning `true`, we ensure vLLM keeps the request in its hash table until
        //      our worker explicitly signals completion, avoiding the race condition.
        //
        //      If the slot is already Finished (no pending operations), we clean it up from our side
        //      but still return `true` so vLLM waits for the worker's completion signal.
        if let SlotState::Finished = slot.state() {
            self.slot_manager().remove_slot(&request_id)?;
        } else {
            debug_assert!(matches!(slot.state(), SlotState::Finishing));
        }

        Ok(true)
    }

    fn has_slot(&self, request_id: String) -> bool {
        self.slot_manager().has_slot(&request_id)
    }

    /// Create a new slot for the given request ID.
    /// This is used to create a new slot for the request.
    fn create_slot(&mut self, request: KvbmRequest, tokens: Vec<u32>) -> anyhow::Result<()> {
        self.slot_manager()
            .create_slot(&request.request_id, tokens, request.salt_hash)?;

        self.inflight_requests.insert(request.request_id);

        Ok(())
    }

    fn reset_cache(&mut self) -> anyhow::Result<bool> {
        match self.slot_manager().reset_prefix_cache() {
            Ok(()) => {
                self.inflight_requests.clear();
                self.onboarding_slots.clear();
                Ok(true)
            }
            Err(error) => {
                tracing::warn!("failed to reset KVBM connector prefix cache: {error:?}");
                Ok(false)
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_speculative_onboarding_notice(
    notice_json: &str,
    actual_request_id: &str,
    actual_worker_id: Option<u64>,
    actual_dp_rank: u32,
    actual_model: &str,
    actual_salt: &str,
    actual_adapter: Option<&str>,
) -> Result<SpeculativeOnboardingNotice, &'static str> {
    let notice: SpeculativeOnboardingNotice =
        serde_json::from_str(notice_json).map_err(|_| "invalid_schema")?;
    if notice.schema != M2_NOTICE_SCHEMA {
        return Err("unsupported_schema");
    }
    let notice_request_id = uuid::Uuid::parse_str(&notice.request_id)
        .map_err(|_| "request_id_mismatch")?
        .to_string();
    let actual_request_id =
        dynamo_runtime::nvtx::canonical_uuid(actual_request_id).ok_or("request_id_mismatch")?;
    if notice_request_id != actual_request_id {
        return Err("request_id_mismatch");
    }
    if actual_worker_id != Some(notice.worker_id) {
        return Err("worker_id_mismatch");
    }
    if actual_dp_rank != notice.dp_rank {
        return Err("dp_rank_mismatch");
    }
    if notice.identity.model != actual_model
        || notice.identity.salt != actual_salt
        || notice.identity.adapter.as_deref() != actual_adapter
    {
        return Err("identity_mismatch");
    }
    if notice.predicted_disk_blocks == 0
        || usize::try_from(notice.predicted_disk_blocks).ok() != Some(notice.block_hashes.len())
    {
        return Err("block_hash_bounds");
    }
    Ok(notice)
}

fn record_speculative_onboarding_notice(
    notices: &mut HashMap<String, SpeculativeOnboardingNotice>,
    notice: SpeculativeOnboardingNotice,
) -> Result<&'static str, &'static str> {
    match notices.get(&notice.request_id) {
        Some(existing) if existing == &notice => Ok("duplicate"),
        Some(_) => Err("conflicting_notice"),
        None => {
            notices.insert(notice.request_id.clone(), notice);
            Ok("accepted")
        }
    }
}

const DYN_M2_DRY_RUN_NOTICE: &str = "DYN_M2_DRY_RUN_NOTICE";
const M2_CONTROL_MAX_BYTES: u64 = 1024 * 1024;

fn m2_dry_run_enabled() -> bool {
    let legacy = std::env::var(DYN_M2_DRY_RUN_NOTICE)
        .ok()
        .is_some_and(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        });
    legacy
        || std::env::var("DYN_M2_POLICY").ok().is_some_and(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "dry_run" | "dry-run" | "naive" | "window_aware" | "window-aware"
            )
        })
}

fn m2_side_effect_enabled() -> bool {
    m2_side_effect_policy().is_some()
}

fn m2_side_effect_policy() -> Option<SpeculativeOnboardingPolicy> {
    match std::env::var("DYN_M2_POLICY")
        .ok()?
        .to_ascii_lowercase()
        .as_str()
    {
        "naive" => Some(SpeculativeOnboardingPolicy::Naive),
        "window_aware" | "window-aware" => Some(SpeculativeOnboardingPolicy::WindowAware),
        _ => None,
    }
}

fn m2_env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn m2_failure_injection() -> Option<String> {
    let qualification = std::env::var("DYN_M2_QUALIFICATION")
        .ok()
        .is_some_and(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true"));
    qualification
        .then(|| std::env::var("DYN_M2_FAILURE_INJECTION").ok())
        .flatten()
}

fn m2_control_socket_path(worker_id: u64, dp_rank: u32) -> PathBuf {
    let directory = std::env::var_os("DYN_M2_CONTROL_SOCKET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    directory.join(format!("dynamo-m2-control-{worker_id}-{dp_rank}.sock"))
}

#[derive(Debug, Serialize)]
struct M2ControlAck<'a> {
    schema: u16,
    status: &'a str,
    disposition: &'a str,
    request_id: &'a str,
    worker_id: u64,
    dp_rank: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<&'a str>,
}

#[derive(Debug)]
struct M2ControlTarget {
    worker_id: u64,
    dp_rank: u32,
    model: String,
    connector_engine_id: String,
}

type M2HostBlocks = Vec<ImmutableBlock<PinnedStorage, VllmLocality, BasicMetadata>>;

enum M2StageState {
    InFlight,
    Ready(M2HostBlocks),
    Demanding(M2HostBlocks),
    Reused,
    Cancelled,
    Expired,
    Failed,
    Rejected,
}

struct M2StageEntry {
    state: M2StageState,
    notice: SpeculativeOnboardingNotice,
    worker_epoch: u64,
    state_version: u64,
    blocks: usize,
    deadline: Instant,
}

#[derive(Default)]
struct M2StageRegistry {
    entries: HashMap<String, M2StageEntry>,
    reserved_blocks: usize,
    in_flight_request: Option<String>,
    worker_epoch: Option<u64>,
    latest_state_version: Option<u64>,
}

pub struct M2StagingManager {
    block_manager: VllmBlockManager,
    runtime: Handle,
    policy: SpeculativeOnboardingPolicy,
    registry: Mutex<M2StageRegistry>,
}

impl std::fmt::Debug for M2StagingManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("M2StagingManager").finish_non_exhaustive()
    }
}

impl M2StagingManager {
    fn new(
        block_manager: VllmBlockManager,
        runtime: Handle,
        policy: SpeculativeOnboardingPolicy,
    ) -> Self {
        Self {
            block_manager,
            runtime,
            policy,
            registry: Mutex::new(M2StageRegistry::default()),
        }
    }

    fn trace(
        notice: &SpeculativeOnboardingNotice,
        event: &str,
        result: &str,
        reason: Option<&str>,
    ) {
        tracing::info!(
            "DYN_M2_TRACE {}",
            serde_json::json!({
                "schema": 1,
                "ts_ns": dynamo_runtime::nvtx::monotonic_ns(),
                "request_id": notice.request_id,
                "request_key": dynamo_runtime::nvtx::request_key(&notice.request_id),
                "component": "staging_manager",
                "event": event,
                "worker_id": notice.worker_id,
                "dp_rank": notice.dp_rank,
                "policy": notice.policy,
                "state_version": notice.state_version,
                "worker_epoch": notice.worker_epoch,
                "state_age_ms": notice.state_age_ms,
                "predicted_blocks": notice.predicted_disk_blocks,
                "bytes": u64::from(notice.predicted_disk_blocks)
                    .saturating_mul(dynamo_runtime::nvtx::PHASE_C_BLOCK_BYTES),
                "deadline_budget_ms": notice.deadline_budget_ms,
                "result": result,
                "reason": reason,
            })
        );
    }

    fn reject(&self, notice: &SpeculativeOnboardingNotice, reason: &'static str) -> &'static str {
        Self::trace(notice, "stage_reject", "rejected", Some(reason));
        Self::trace(notice, "fallback", "passive", Some(reason));
        if let Ok(mut registry) = self.registry.lock() {
            registry.entries.insert(
                notice.request_id.clone(),
                M2StageEntry {
                    state: M2StageState::Rejected,
                    notice: notice.clone(),
                    worker_epoch: notice.worker_epoch,
                    state_version: notice.state_version,
                    blocks: 0,
                    deadline: Instant::now(),
                },
            );
        }
        "rejected"
    }

    fn admit(self: &Arc<Self>, notice: SpeculativeOnboardingNotice) -> &'static str {
        if notice.policy != self.policy {
            return self.reject(&notice, "policy_mismatch");
        }
        if notice.state_age_ms > m2_env_u64("DYN_M2_FRESHNESS_MS", 600_000) {
            return self.reject(&notice, "state_stale");
        }
        match m2_failure_injection().as_deref() {
            Some("identity") => return self.reject(&notice, "identity_injection"),
            Some("g2_full") => return self.reject(&notice, "g2_full_injection"),
            _ => {}
        }

        let block_count = notice.block_hashes.len();
        let deadline =
            Instant::now() + Duration::from_millis(notice.deadline_budget_ms.unwrap_or(1).max(1));
        let Some(host_pool) = self.block_manager.host() else {
            return self.reject(&notice, "g2_unavailable");
        };
        let Some(disk_pool) = self.block_manager.disk() else {
            return self.reject(&notice, "g3_unavailable");
        };
        let capacity_limit = usize::try_from(host_pool.total_blocks() / 4).unwrap_or(usize::MAX);
        if block_count == 0
            || block_count > capacity_limit
            || u64::try_from(block_count).unwrap_or(u64::MAX) > host_pool.available_blocks()
        {
            return self.reject(&notice, "g2_capacity");
        }

        {
            let Ok(mut registry) = self.registry.lock() else {
                return "rejected";
            };
            if let Some(existing) = registry.entries.get(&notice.request_id) {
                if existing.worker_epoch == notice.worker_epoch
                    && existing.state_version == notice.state_version
                {
                    return "duplicate";
                }
                return "rejected";
            }
            if registry
                .worker_epoch
                .is_some_and(|epoch| epoch != notice.worker_epoch)
            {
                drop(registry);
                return self.reject(&notice, "worker_epoch_mismatch");
            }
            if registry
                .latest_state_version
                .is_some_and(|version| notice.state_version < version)
            {
                drop(registry);
                return self.reject(&notice, "state_version_regression");
            }
            if registry.in_flight_request.is_some() {
                drop(registry);
                return self.reject(&notice, "in_flight_limit");
            }
            registry.worker_epoch = Some(notice.worker_epoch);
            registry.latest_state_version = Some(notice.state_version);
            registry.reserved_blocks = registry.reserved_blocks.saturating_add(block_count);
            registry.in_flight_request = Some(notice.request_id.clone());
            registry.entries.insert(
                notice.request_id.clone(),
                M2StageEntry {
                    state: M2StageState::InFlight,
                    notice: notice.clone(),
                    worker_epoch: notice.worker_epoch,
                    state_version: notice.state_version,
                    blocks: block_count,
                    deadline,
                },
            );
        }

        let disk_blocks = match disk_pool.match_sequence_hashes_blocking(&notice.block_hashes) {
            Ok(blocks) if blocks.len() == block_count => blocks,
            _ => return self.fail_admission(&notice, block_count, "disk_match_mismatch"),
        };
        let targets = match host_pool.allocate_blocks_blocking(block_count) {
            Ok(blocks) => blocks,
            Err(_) => return self.fail_admission(&notice, block_count, "g2_reservation_failed"),
        };
        let still_active = self.registry.lock().is_ok_and(|registry| {
            registry.in_flight_request.as_deref() == Some(&notice.request_id)
                && registry
                    .entries
                    .get(&notice.request_id)
                    .is_some_and(|entry| matches!(entry.state, M2StageState::InFlight))
        });
        if !still_active {
            drop(targets);
            return self.fail_admission(&notice, block_count, "cancelled_before_submit");
        }

        Self::trace(&notice, "stage_reserve", "reserved", None);
        let receiver = self
            .block_manager
            .stage_disk_blocks(disk_blocks, Some(targets));
        Self::trace(&notice, "stage_submit", "submitted", None);
        Self::trace(&notice, "stage_transfer_start", "in_flight", None);

        let manager = self.clone();
        let transfer_notice = notice.clone();
        self.runtime.spawn(async move {
            if m2_failure_injection().as_deref() == Some("cancel") {
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
            let result = receiver.await;
            let inject_transfer_failure =
                m2_failure_injection().as_deref() == Some("transfer_failure");
            let mut release_reason = Some("orphaned_after_reset");
            if let Ok(mut registry) = manager.registry.lock() {
                if registry.in_flight_request.as_deref() == Some(&transfer_notice.request_id) {
                    registry.in_flight_request = None;
                }
                let mut released = block_count;
                if let Some(entry) = registry.entries.get_mut(&transfer_notice.request_id) {
                    match (&entry.state, result) {
                        (M2StageState::InFlight, Ok(Ok(blocks)))
                            if !inject_transfer_failure && Instant::now() <= entry.deadline =>
                        {
                            entry.state = M2StageState::Ready(blocks);
                            released = 0;
                            release_reason = None;
                        }
                        (M2StageState::InFlight, Ok(Ok(_))) if inject_transfer_failure => {
                            entry.state = M2StageState::Failed;
                            released = entry.blocks;
                            release_reason = Some("transfer_failure_injection");
                        }
                        (M2StageState::InFlight, Ok(Ok(_))) => {
                            entry.state = M2StageState::Expired;
                            released = entry.blocks;
                            release_reason = Some("deadline");
                        }
                        (M2StageState::InFlight, _) => {
                            entry.state = M2StageState::Failed;
                            released = entry.blocks;
                            release_reason = Some("transfer_failure");
                        }
                        (_, Ok(Ok(_))) => {
                            released = entry.blocks;
                            release_reason = Some("cancelled_or_expired");
                        }
                        (_, _) => {
                            released = entry.blocks;
                            release_reason = Some("drained_failure");
                        }
                    }
                }
                registry.reserved_blocks = registry.reserved_blocks.saturating_sub(released);
            }
            Self::trace(
                &transfer_notice,
                "stage_transfer_end",
                if release_reason.is_none() {
                    "ok"
                } else {
                    "released"
                },
                release_reason,
            );
            if release_reason == Some("deadline") {
                Self::trace(&transfer_notice, "timeout", "expired", Some("deadline"));
            }
            if release_reason.is_none() {
                Self::trace(&transfer_notice, "staged_ready", "ready", None);
            } else {
                Self::trace(&transfer_notice, "fallback", "passive", release_reason);
                Self::trace(&transfer_notice, "release", "released", release_reason);
            }
        });

        if m2_failure_injection().as_deref() == Some("reset") {
            let manager = self.clone();
            self.runtime.spawn(async move {
                tokio::time::sleep(Duration::from_millis(1)).await;
                manager.reset();
            });
        }

        let manager = self.clone();
        let deadline_notice = notice;
        self.runtime.spawn(async move {
            tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await;
            let mut released = 0;
            if let Ok(mut registry) = manager.registry.lock()
                && let Some(entry) = registry.entries.get_mut(&deadline_notice.request_id)
                && matches!(entry.state, M2StageState::Ready(_))
            {
                entry.state = M2StageState::Expired;
                released = entry.blocks;
                registry.reserved_blocks = registry.reserved_blocks.saturating_sub(released);
            }
            if released > 0 {
                Self::trace(&deadline_notice, "timeout", "expired", Some("deadline"));
                Self::trace(&deadline_notice, "fallback", "passive", Some("deadline"));
                Self::trace(&deadline_notice, "release", "released", Some("deadline"));
            }
        });
        "submitted"
    }

    fn fail_admission(
        &self,
        notice: &SpeculativeOnboardingNotice,
        block_count: usize,
        reason: &'static str,
    ) -> &'static str {
        if let Ok(mut registry) = self.registry.lock() {
            if registry.in_flight_request.as_deref() == Some(&notice.request_id) {
                registry.in_flight_request = None;
            }
            registry.reserved_blocks = registry.reserved_blocks.saturating_sub(block_count);
            if let Some(entry) = registry.entries.get_mut(&notice.request_id) {
                entry.state = M2StageState::Rejected;
            }
        }
        Self::trace(notice, "stage_reject", "rejected", Some(reason));
        Self::trace(notice, "fallback", "passive", Some(reason));
        "rejected"
    }

    fn demand(&self, request_id: &str) -> bool {
        let mut traces = Vec::new();
        let ready = if let Ok(mut registry) = self.registry.lock() {
            match registry.entries.get_mut(request_id) {
                Some(entry) => match &mut entry.state {
                    M2StageState::Ready(blocks) => {
                        let held = std::mem::take(blocks);
                        entry.state = M2StageState::Demanding(held);
                        true
                    }
                    M2StageState::InFlight => {
                        entry.state = M2StageState::Cancelled;
                        traces.push((
                            entry.notice.clone(),
                            "cancel",
                            "cancelled",
                            Some("demand_before_ready"),
                        ));
                        traces.push((
                            entry.notice.clone(),
                            "fallback",
                            "passive",
                            Some("demand_before_ready"),
                        ));
                        false
                    }
                    _ => false,
                },
                None => false,
            }
        } else {
            false
        };
        for (notice, event, result, reason) in traces {
            Self::trace(&notice, event, result, reason);
        }
        ready
    }

    fn complete_demand(&self, request_id: &str, matched: bool) {
        let mut trace = None;
        if let Ok(mut registry) = self.registry.lock()
            && let Some(entry) = registry.entries.get_mut(request_id)
            && matches!(entry.state, M2StageState::Demanding(_))
        {
            entry.state = if matched {
                M2StageState::Reused
            } else {
                M2StageState::Cancelled
            };
            let blocks = entry.blocks;
            trace = Some((entry.notice.clone(), matched));
            registry.reserved_blocks = registry.reserved_blocks.saturating_sub(blocks);
        }
        if let Some((notice, matched)) = trace {
            if matched {
                Self::trace(&notice, "demand_reuse", "reused", None);
            } else {
                Self::trace(&notice, "fallback", "passive", Some("staged_match_miss"));
                Self::trace(&notice, "release", "released", Some("staged_match_miss"));
            }
        }
    }

    fn cancel(&self, request_id: &str, reason: &'static str) {
        let mut trace = None;
        if let Ok(mut registry) = self.registry.lock()
            && let Some(entry) = registry.entries.get_mut(request_id)
        {
            let released = match &mut entry.state {
                M2StageState::Ready(blocks) | M2StageState::Demanding(blocks) => {
                    blocks.clear();
                    entry.blocks
                }
                M2StageState::InFlight => 0,
                _ => return,
            };
            entry.state = M2StageState::Cancelled;
            trace = Some((entry.notice.clone(), released > 0));
            registry.reserved_blocks = registry.reserved_blocks.saturating_sub(released);
        }
        if let Some((notice, released)) = trace {
            Self::trace(&notice, "cancel", "cancelled", Some(reason));
            if released {
                Self::trace(&notice, "release", "released", Some(reason));
            }
        }
    }

    fn reset(&self) {
        let mut released = Vec::new();
        if let Ok(mut registry) = self.registry.lock() {
            let in_flight_blocks = registry
                .in_flight_request
                .as_ref()
                .and_then(|request_id| registry.entries.get(request_id))
                .map_or(0, |entry| entry.blocks);
            released.extend(
                registry
                    .entries
                    .values()
                    .filter(|entry| {
                        matches!(
                            entry.state,
                            M2StageState::Ready(_) | M2StageState::Demanding(_)
                        )
                    })
                    .map(|entry| entry.notice.clone()),
            );
            registry.entries.clear();
            registry.reserved_blocks = in_flight_blocks;
            registry.worker_epoch = None;
            registry.latest_state_version = None;
        }
        for notice in released {
            Self::trace(&notice, "cancel", "cancelled", Some("reset"));
            Self::trace(&notice, "fallback", "passive", Some("reset"));
            Self::trace(&notice, "release", "released", Some("reset"));
        }
    }
}

fn observe_m2_control_notice(
    notice_json: &str,
    target: &M2ControlTarget,
    notices: &Mutex<HashMap<String, SpeculativeOnboardingNotice>>,
    staging: Option<&Arc<M2StagingManager>>,
) -> String {
    let notice = match serde_json::from_str::<SpeculativeOnboardingNotice>(notice_json) {
        Ok(notice) => notice,
        Err(_) => {
            return serde_json::to_string(&M2ControlAck {
                schema: M2_NOTICE_SCHEMA,
                status: "ok",
                disposition: "rejected",
                request_id: "",
                worker_id: target.worker_id,
                dp_rank: target.dp_rank,
                reason: Some("invalid_schema"),
            })
            .expect("M2 control rejection is serializable");
        }
    };
    let reject = |reason: &'static str| {
        serde_json::to_string(&M2ControlAck {
            schema: M2_NOTICE_SCHEMA,
            status: "ok",
            disposition: "rejected",
            request_id: &notice.request_id,
            worker_id: target.worker_id,
            dp_rank: target.dp_rank,
            reason: Some(reason),
        })
        .expect("M2 control rejection is serializable")
    };
    if notice.schema != M2_NOTICE_SCHEMA {
        return reject("unsupported_schema");
    }
    if uuid::Uuid::parse_str(&notice.request_id).is_err() {
        return reject("request_id_mismatch");
    }
    if notice.worker_id != target.worker_id {
        return reject("worker_id_mismatch");
    }
    if notice.dp_rank != target.dp_rank {
        return reject("dp_rank_mismatch");
    }
    if notice.identity.model != target.model {
        return reject("identity_mismatch");
    }
    if !notice.identity.salt.is_empty() || notice.identity.adapter.is_some() {
        return reject("unsupported_identity");
    }
    if notice.predicted_disk_blocks == 0
        || usize::try_from(notice.predicted_disk_blocks).ok() != Some(notice.block_hashes.len())
    {
        return reject("block_hash_bounds");
    }

    let registry_disposition = match notices.lock() {
        Ok(mut notices) => match record_speculative_onboarding_notice(&mut notices, notice.clone())
        {
            Ok(disposition) => disposition,
            Err(reason) => return reject(reason),
        },
        Err(_) => return reject("registry_poisoned"),
    };
    let (disposition, admission_reason) = if registry_disposition == "accepted" {
        match (notice.policy, staging) {
            (SpeculativeOnboardingPolicy::DryRun, _) => (registry_disposition, None),
            (_, Some(manager)) => (manager.admit(notice.clone()), None),
            (_, None) => ("rejected", Some("staging_unavailable")),
        }
    } else {
        (registry_disposition, None)
    };
    tracing::info!(
        "DYN_M2_TRACE {}",
        serde_json::json!({
            "schema": 1,
            "ts_ns": dynamo_runtime::nvtx::monotonic_ns(),
            "request_id": notice.request_id,
            "component": "connector",
            "event": "notice_received",
            "transport": "addressed_control",
            "worker_id": notice.worker_id,
            "dp_rank": notice.dp_rank,
            "connector_engine_id": target.connector_engine_id,
            "state_version": notice.state_version,
            "worker_epoch": notice.worker_epoch,
            "predicted_disk_blocks": notice.predicted_disk_blocks,
            "block_hashes": &notice.block_hashes,
            "disposition": disposition,
        })
    );
    serde_json::to_string(&M2ControlAck {
        schema: M2_NOTICE_SCHEMA,
        status: "ok",
        disposition,
        request_id: &notice.request_id,
        worker_id: target.worker_id,
        dp_rank: target.dp_rank,
        reason: admission_reason,
    })
    .expect("M2 control acknowledgement is serializable")
}

async fn run_m2_control_blocking<F>(operation: F) -> Result<String, tokio::task::JoinError>
where
    F: FnOnce() -> String + Send + 'static,
{
    tokio::task::spawn_blocking(operation).await
}

#[derive(Debug)]
struct M2ControlServer {
    cancel: CancellationToken,
    socket_path: PathBuf,
}

impl M2ControlServer {
    fn start(
        handle: &Handle,
        target: M2ControlTarget,
        notices: Arc<Mutex<HashMap<String, SpeculativeOnboardingNotice>>>,
        staging: Option<Arc<M2StagingManager>>,
    ) -> anyhow::Result<Self> {
        let socket_path = m2_control_socket_path(target.worker_id, target.dp_rank);
        Self::start_at(handle, target, notices, staging, socket_path)
    }

    fn start_at(
        handle: &Handle,
        target: M2ControlTarget,
        notices: Arc<Mutex<HashMap<String, SpeculativeOnboardingNotice>>>,
        staging: Option<Arc<M2StagingManager>>,
        socket_path: PathBuf,
    ) -> anyhow::Result<Self> {
        if Path::new(&socket_path).exists() {
            if std::os::unix::net::UnixStream::connect(&socket_path).is_ok() {
                anyhow::bail!(
                    "M2 control socket is already active: {}",
                    socket_path.display()
                );
            }
            std::fs::remove_file(&socket_path)?;
        }
        let listener = {
            let _runtime_guard = handle.enter();
            UnixListener::bind(&socket_path)?
        };
        let cancel = CancellationToken::new();
        let task_cancel = cancel.clone();
        let target = Arc::new(target);
        handle.spawn(async move {
            loop {
                let (stream, _) = tokio::select! {
                    _ = task_cancel.cancelled() => break,
                    accepted = listener.accept() => match accepted {
                        Ok(accepted) => accepted,
                        Err(error) => {
                            tracing::error!(%error, "M2 control accept failed");
                            break;
                        }
                    },
                };
                let target = target.clone();
                let notices = notices.clone();
                let staging = staging.clone();
                tokio::spawn(async move {
                    let (reader, mut writer) = stream.into_split();
                    let mut reader = BufReader::new(reader).take(M2_CONTROL_MAX_BYTES + 1);
                    let mut request = Vec::new();
                    let result = reader.read_until(b'\n', &mut request).await;
                    let response = match result {
                        Ok(read)
                            if read > 0
                                && read as u64 <= M2_CONTROL_MAX_BYTES
                                && request.last() == Some(&b'\n') =>
                        {
                            request.pop();
                            match std::str::from_utf8(&request) {
                                Ok(request) => {
                                    let request = request.to_string();
                                    match run_m2_control_blocking(move || {
                                        observe_m2_control_notice(
                                            &request,
                                            &target,
                                            &notices,
                                            staging.as_ref(),
                                        )
                                    })
                                    .await
                                    {
                                        Ok(response) => response,
                                        Err(error) => {
                                            tracing::warn!(
                                                %error,
                                                "M2 control admission task failed closed"
                                            );
                                            serde_json::json!({
                                                "schema": M2_NOTICE_SCHEMA,
                                                "status": "ok",
                                                "disposition": "rejected",
                                                "reason": "control_unavailable",
                                            })
                                            .to_string()
                                        }
                                    }
                                }
                                Err(_) => serde_json::json!({
                                    "schema": M2_NOTICE_SCHEMA,
                                    "status": "ok",
                                    "disposition": "rejected",
                                    "reason": "invalid_utf8",
                                })
                                .to_string(),
                            }
                        }
                        _ => serde_json::json!({
                            "schema": M2_NOTICE_SCHEMA,
                            "status": "ok",
                            "disposition": "rejected",
                            "reason": "invalid_frame",
                        })
                        .to_string(),
                    };
                    if let Err(error) = writer.write_all(response.as_bytes()).await {
                        tracing::warn!(%error, "M2 control acknowledgement write failed");
                        return;
                    }
                    let _ = writer.write_all(b"\n").await;
                });
            }
        });
        Ok(Self {
            cancel,
            socket_path,
        })
    }
}

impl Drop for M2ControlServer {
    fn drop(&mut self) {
        self.cancel.cancel();
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

#[pyclass]
pub struct PyKvConnectorLeader {
    connector_leader: Box<dyn Leader>,
    m2_dry_run_notices: Arc<Mutex<HashMap<String, SpeculativeOnboardingNotice>>>,
    m2_staging_manager: Option<Arc<M2StagingManager>>,
    _m2_control_server: Option<M2ControlServer>,
}

#[pymethods]
impl PyKvConnectorLeader {
    #[new]
    #[pyo3(signature = (worker_id, drt, page_size, leader, consolidator_vllm_endpoint=None, consolidator_output_endpoint=None, consolidator_mode=None, m2_worker_id=None, m2_dp_rank=None, m2_model=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        worker_id: String,
        drt: Option<PyObject>,
        page_size: usize,
        leader: PyKvbmLeader,
        consolidator_vllm_endpoint: Option<String>,
        consolidator_output_endpoint: Option<String>,
        consolidator_mode: Option<String>,
        m2_worker_id: Option<u64>,
        m2_dp_rank: Option<u32>,
        m2_model: Option<String>,
    ) -> PyResult<Self> {
        let _ = &drt; // drt is currently un-used in leader

        // Initialize logging for the vLLM connector
        dynamo_runtime::logging::init();

        let enable_kvbm_record = std::env::var(env_kvbm::DYN_KVBM_ENABLE_RECORD)
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let connector_engine_id = worker_id.clone();
        let connector_leader: Box<dyn Leader> = if enable_kvbm_record {
            Box::new(recorder::KvConnectorLeaderRecorder::new(
                worker_id,
                page_size,
                leader,
                consolidator_vllm_endpoint,
                consolidator_output_endpoint,
                consolidator_mode,
            ))
        } else {
            Box::new(KvConnectorLeader::new(
                worker_id,
                page_size,
                leader,
                consolidator_vllm_endpoint,
                consolidator_output_endpoint,
                consolidator_mode,
            ))
        };
        let m2_staging_manager = connector_leader.m2_staging_manager();
        let m2_dry_run_notices = Arc::new(Mutex::new(HashMap::new()));
        let m2_control_server = if m2_dry_run_enabled() {
            match (m2_worker_id, m2_dp_rank, m2_model) {
                (Some(worker_id), Some(dp_rank), Some(model)) => M2ControlServer::start(
                    &get_current_tokio_handle(),
                    M2ControlTarget {
                        worker_id,
                        dp_rank,
                        model,
                        connector_engine_id,
                    },
                    m2_dry_run_notices.clone(),
                    m2_staging_manager.clone(),
                )
                .map(Some)
                .unwrap_or_else(|error| {
                    tracing::error!(%error, "M2 control server failed to start; falling back passive");
                    None
                }),
                _ => {
                    tracing::warn!("M2 dry-run enabled without a complete connector target; control disabled");
                    None
                }
            }
        } else {
            None
        };
        Ok(Self {
            connector_leader,
            m2_dry_run_notices,
            m2_staging_manager,
            _m2_control_server: m2_control_server,
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        notice_json,
        actual_request_id,
        actual_worker_id,
        actual_dp_rank,
        actual_model,
        actual_salt,
        actual_adapter,
        connector_engine_id
    ))]
    fn observe_speculative_onboarding_notice(
        &mut self,
        notice_json: &str,
        actual_request_id: &str,
        actual_worker_id: Option<u64>,
        actual_dp_rank: u32,
        actual_model: &str,
        actual_salt: &str,
        actual_adapter: Option<String>,
        connector_engine_id: &str,
    ) -> String {
        let reject = |reason: &str| {
            tracing::warn!(
                "DYN_M2_TRACE {}",
                serde_json::json!({
                    "schema": 1,
                    "ts_ns": dynamo_runtime::nvtx::monotonic_ns(),
                    "request_id": actual_request_id,
                    "component": "connector",
                    "event": "notice_received",
                    "connector_engine_id": connector_engine_id,
                    "disposition": "rejected",
                    "reason": reason,
                })
            );
            format!("rejected:{reason}")
        };

        let notice = match validate_speculative_onboarding_notice(
            notice_json,
            actual_request_id,
            actual_worker_id,
            actual_dp_rank,
            actual_model,
            actual_salt,
            actual_adapter.as_deref(),
        ) {
            Ok(notice) => notice,
            Err(reason) => return reject(reason),
        };

        let disposition = match self.m2_dry_run_notices.lock() {
            Ok(mut notices) => {
                match record_speculative_onboarding_notice(&mut notices, notice.clone()) {
                    Ok(disposition) => disposition,
                    Err(reason) => return reject(reason),
                }
            }
            Err(_) => return reject("registry_poisoned"),
        };
        tracing::info!(
            "DYN_M2_TRACE {}",
            serde_json::json!({
                "schema": 1,
                "ts_ns": dynamo_runtime::nvtx::monotonic_ns(),
                "request_id": notice.request_id,
                "engine_request_id": actual_request_id,
                "component": "connector",
                "event": "notice_received",
                "worker_id": notice.worker_id,
                "dp_rank": notice.dp_rank,
                "connector_engine_id": connector_engine_id,
                "state_version": notice.state_version,
                "worker_epoch": notice.worker_epoch,
                "predicted_disk_blocks": notice.predicted_disk_blocks,
                "block_hashes": &notice.block_hashes,
                "disposition": disposition,
            })
        );
        disposition.to_string()
    }

    fn get_num_new_matched_tokens(
        &self,
        request_id: String,
        request_num_tokens: usize,
        num_computed_tokens: usize,
    ) -> PyResult<(usize, bool)> {
        let canonical_request_id = dynamo_runtime::nvtx::canonical_uuid(&request_id);
        let staged_ready = if let Some(canonical_request_id) = canonical_request_id.as_deref()
            && self
                .m2_dry_run_notices
                .lock()
                .is_ok_and(|notices| notices.contains_key(canonical_request_id))
        {
            tracing::info!(
                "DYN_M2_TRACE {}",
                serde_json::json!({
                    "schema": 1,
                    "ts_ns": dynamo_runtime::nvtx::monotonic_ns(),
                    "request_id": canonical_request_id,
                    "engine_request_id": request_id,
                    "component": "connector",
                    "event": "connector_match_start",
                })
            );
            self.m2_staging_manager
                .as_ref()
                .is_some_and(|manager| manager.demand(canonical_request_id))
        } else {
            false
        };
        let result = self.connector_leader.get_num_new_matched_tokens(
            request_id,
            request_num_tokens,
            num_computed_tokens,
        );
        if staged_ready
            && let (Some(manager), Some(canonical_request_id)) = (
                self.m2_staging_manager.as_ref(),
                canonical_request_id.as_deref(),
            )
        {
            let matched = result
                .as_ref()
                .is_ok_and(|(tokens, has_match)| *has_match && *tokens > 0);
            manager.complete_demand(canonical_request_id, matched);
        }
        result.map_err(to_pyerr)
    }

    fn update_state_after_alloc(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> PyResult<()> {
        self.connector_leader
            .update_state_after_alloc(request_id, block_ids, num_external_tokens)
            .map_err(to_pyerr)
    }

    fn build_connector_metadata(&mut self, scheduler_output: SchedulerOutput) -> PyResult<Vec<u8>> {
        self.connector_leader
            .build_connector_metadata(scheduler_output)
            .map_err(to_pyerr)
    }

    fn request_finished(&mut self, request_id: &str, block_ids: Vec<BlockId>) -> PyResult<bool> {
        let result = self
            .connector_leader
            .request_finished(request_id.to_string(), block_ids)
            .map_err(to_pyerr);
        if let Ok(mut notices) = self.m2_dry_run_notices.lock() {
            let canonical = dynamo_runtime::nvtx::canonical_uuid(request_id)
                .unwrap_or_else(|| request_id.to_string());
            notices.remove(&canonical);
        }
        if let Some(manager) = self.m2_staging_manager.as_ref() {
            let canonical = dynamo_runtime::nvtx::canonical_uuid(request_id)
                .unwrap_or_else(|| request_id.to_string());
            manager.cancel(&canonical, "request_finished");
        }
        result
    }

    fn has_slot(&self, request_id: &str) -> bool {
        self.connector_leader.has_slot(request_id.to_string())
    }

    fn create_slot(&mut self, request: KvbmRequest, tokens: Vec<u32>) -> PyResult<()> {
        self.connector_leader
            .create_slot(request, tokens)
            .map_err(to_pyerr)
    }

    fn reset_cache(&mut self, py: Python<'_>) -> PyResult<bool> {
        let reset = py
            .allow_threads(|| self.connector_leader.reset_cache())
            .map_err(to_pyerr)?;
        if reset && let Ok(mut notices) = self.m2_dry_run_notices.lock() {
            notices.clear();
        }
        if reset && let Some(manager) = self.m2_staging_manager.as_ref() {
            manager.reset();
        }
        Ok(reset)
    }
}

pub fn kvbm_metrics_endpoint_enabled() -> bool {
    std::env::var(env_kvbm::DYN_KVBM_METRICS)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub fn parse_kvbm_metrics_port() -> u16 {
    match std::env::var(env_kvbm::DYN_KVBM_METRICS_PORT) {
        Ok(val) => match val.trim().parse::<u16>() {
            Ok(port) => port,
            Err(_) => {
                tracing::warn!(
                    "[kvbm] Invalid DYN_KVBM_METRICS_PORT='{}', falling back to 6880",
                    val
                );
                6880
            }
        },
        Err(_) => {
            tracing::warn!(
                "DYN_KVBM_METRICS_PORT not present or couldn’t be interpreted, falling back to 6880"
            );
            6880
        }
    }
}

#[cfg(test)]
mod m2_notice_tests {
    use super::*;
    use dynamo_llm::kv_router::scheduling::{
        SpeculativeCacheIdentity, SpeculativeOnboardingPolicy,
    };

    fn notice(request_id: &str) -> SpeculativeOnboardingNotice {
        SpeculativeOnboardingNotice {
            schema: M2_NOTICE_SCHEMA,
            request_id: request_id.to_string(),
            worker_id: 7,
            dp_rank: 0,
            identity: SpeculativeCacheIdentity {
                model: "model".to_string(),
                salt: String::new(),
                adapter: None,
            },
            worker_epoch: 2,
            state_version: 3,
            state_age_ms: 4,
            prefix_start_block: 1,
            block_hashes: vec![11, 12],
            predicted_disk_blocks: 2,
            predicted_queue_window_ms: None,
            estimated_stage_ms: None,
            deadline_budget_ms: None,
            policy: SpeculativeOnboardingPolicy::DryRun,
        }
    }

    const REQUEST_ID: &str = "11111111-1111-4111-8111-111111111111";

    fn validate(value: &str) -> Result<SpeculativeOnboardingNotice, &'static str> {
        validate_speculative_onboarding_notice(value, REQUEST_ID, Some(7), 0, "model", "", None)
    }

    #[test]
    fn validates_schema_target_identity_and_hash_bounds() {
        let valid = serde_json::to_string(&notice(REQUEST_ID)).unwrap();
        assert_eq!(validate(&valid), Ok(notice(REQUEST_ID)));
        assert_eq!(
            validate_speculative_onboarding_notice(
                &valid,
                &format!("{REQUEST_ID}-bdc779c6"),
                Some(7),
                0,
                "model",
                "",
                None,
            ),
            Ok(notice(REQUEST_ID))
        );
        assert_eq!(validate("{}"), Err("invalid_schema"));

        let mut wrong_request = notice("22222222-2222-4222-8222-222222222222");
        assert_eq!(
            validate(&serde_json::to_string(&wrong_request).unwrap()),
            Err("request_id_mismatch")
        );
        wrong_request.request_id = REQUEST_ID.to_string();
        wrong_request.identity.model = "other-model".to_string();
        assert_eq!(
            validate(&serde_json::to_string(&wrong_request).unwrap()),
            Err("identity_mismatch")
        );

        let mut bad_bounds = notice(REQUEST_ID);
        bad_bounds.predicted_disk_blocks = 1;
        assert_eq!(
            validate(&serde_json::to_string(&bad_bounds).unwrap()),
            Err("block_hash_bounds")
        );
    }

    #[test]
    fn missing_or_wrong_dispatch_target_fails_closed() {
        let value = serde_json::to_string(&notice(REQUEST_ID)).unwrap();
        assert_eq!(
            validate_speculative_onboarding_notice(&value, REQUEST_ID, None, 0, "model", "", None,),
            Err("worker_id_mismatch")
        );
        assert_eq!(
            validate_speculative_onboarding_notice(
                &value,
                REQUEST_ID,
                Some(7),
                1,
                "model",
                "",
                None,
            ),
            Err("dp_rank_mismatch")
        );
    }

    #[test]
    fn duplicate_is_idempotent_and_conflict_rejects() {
        let mut notices = HashMap::new();
        assert_eq!(
            record_speculative_onboarding_notice(&mut notices, notice(REQUEST_ID)),
            Ok("accepted")
        );
        assert_eq!(
            record_speculative_onboarding_notice(&mut notices, notice(REQUEST_ID)),
            Ok("duplicate")
        );
        let mut conflict = notice(REQUEST_ID);
        conflict.state_version += 1;
        assert_eq!(
            record_speculative_onboarding_notice(&mut notices, conflict),
            Err("conflicting_notice")
        );
        assert_eq!(notices.len(), 1);
    }

    #[test]
    fn side_effect_notice_without_staging_manager_fails_closed() {
        let mut side_effect = notice(REQUEST_ID);
        side_effect.policy = SpeculativeOnboardingPolicy::Naive;
        let notices = Mutex::new(HashMap::new());
        let response = observe_m2_control_notice(
            &serde_json::to_string(&side_effect).unwrap(),
            &M2ControlTarget {
                worker_id: 7,
                dp_rank: 0,
                model: "model".to_string(),
                connector_engine_id: "engine".to_string(),
            },
            &notices,
            None,
        );
        let response: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(response["disposition"], "rejected");
    }

    #[test]
    fn unsupported_cache_identity_fails_closed_at_control_endpoint() {
        let mut unsupported = notice(REQUEST_ID);
        unsupported.identity.salt = "tenant".to_string();
        let notices = Mutex::new(HashMap::new());
        let response = observe_m2_control_notice(
            &serde_json::to_string(&unsupported).unwrap(),
            &M2ControlTarget {
                worker_id: 7,
                dp_rank: 0,
                model: "model".to_string(),
                connector_engine_id: "engine".to_string(),
            },
            &notices,
            None,
        );
        let response: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(response["disposition"], "rejected");
        assert_eq!(response["reason"], "unsupported_identity");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn control_admission_runs_on_blocking_executor() {
        let (sender, receiver) = oneshot::channel();
        sender.send("accepted").unwrap();

        let response = run_m2_control_blocking(move || {
            receiver
                .blocking_recv()
                .expect("blocking admission input remains available")
                .to_string()
        })
        .await
        .expect("blocking admission task succeeds");

        assert_eq!(response, "accepted");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn addressed_control_socket_acknowledges_before_demand() {
        let socket_path = PathBuf::from(format!(
            "/tmp/dynamo-m2-control-test-{}.sock",
            uuid::Uuid::new_v4()
        ));
        let notices = Arc::new(Mutex::new(HashMap::new()));
        let server = M2ControlServer::start_at(
            &Handle::current(),
            M2ControlTarget {
                worker_id: 7,
                dp_rank: 0,
                model: "model".to_string(),
                connector_engine_id: "engine".to_string(),
            },
            notices.clone(),
            None,
            socket_path.clone(),
        )
        .unwrap();

        let mut stream = tokio::net::UnixStream::connect(&socket_path).await.unwrap();
        let request = serde_json::to_vec(&notice(REQUEST_ID)).unwrap();
        stream.write_all(&request).await.unwrap();
        stream.write_all(b"\n").await.unwrap();
        let mut response = String::new();
        BufReader::new(stream)
            .read_line(&mut response)
            .await
            .unwrap();
        let response: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(response["status"], "ok");
        assert_eq!(response["disposition"], "accepted");
        assert!(notices.lock().unwrap().contains_key(REQUEST_ID));

        drop(server);
        assert!(!socket_path.exists());
    }
}
