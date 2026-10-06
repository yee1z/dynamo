// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use super::staged::{
    DISK_BOUNCE_FAIL_READS_ENV, DiskBounce, DiskBounceConfig, Segment, SegmentIo, StagedEvent,
    plan_segments, run_staged,
};
use super::zmq::*;
use futures::future::try_join_all;
use nixl_sys::NixlDescriptor;
use utils::*;

use BlockTransferPool::*;

use crate::block_manager::{
    BasicMetadata, Storage,
    block::{
        Block, BlockDataProvider, BlockDataProviderMut, ReadableBlock, WritableBlock,
        data::local::LocalBlockData,
        locality,
        transfer::{TransferContext, WriteTo, WriteToStrategy, read_disk_blocks_into_host},
    },
    connector::protocol::TransferBlocksFailed,
    connector::scheduler::{SchedulingDecision, TransferSchedulerClient},
    offload::max_transfer_batch_size,
    storage::{DeviceStorage, DiskStorage, Local, PinnedStorage},
};

use anyhow::Result;
use async_trait::async_trait;
use std::{
    any::Any,
    sync::{Arc, OnceLock},
};

use dynamo_runtime::nvtx::{
    self, CATEGORY_TRANSFER, PHASE_C_BLOCK_BYTES, PhaseCPayload, PhaseCRange,
};

fn m1_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("DYN_M1_TRACE").is_some())
}

#[allow(clippy::too_many_arguments)]
fn emit_transfer_boundary(
    event: &str,
    request_id: &str,
    request_key: u64,
    transfer_id: &str,
    transfer_key: u64,
    direction: &str,
    tier_code: u64,
    blocks: u64,
    bytes: u64,
    status: Option<&str>,
) {
    if !m1_trace_enabled() {
        return;
    }
    tracing::info!(
        "DYN_M1_TRACE {}",
        serde_json::json!({
            "schema": 1,
            "ts_ns": nvtx::monotonic_ns(),
            "request_id": request_id,
            "request_key": request_key,
            "request_key_hex": nvtx::key_hex(request_key),
            "transfer_id": transfer_id,
            "transfer_key": transfer_key,
            "transfer_key_hex": nvtx::key_hex(transfer_key),
            "component": "physical",
            "event": event,
            "direction": direction,
            "tier_code": tier_code,
            "blocks": blocks,
            "bytes": bytes,
            "status": status,
        })
    );
}

/// Identity of an onboarding transfer for the per-segment trace events of the staged path.
struct TransferTrace {
    request_id: String,
    request_key: u64,
    transfer_id: String,
    transfer_key: u64,
    direction: &'static str,
    tier_code: u64,
    bytes_per_block: u64,
}

impl TransferTrace {
    /// `disk_read_start/end` and `bounce_copy_start/end` of one segment (trace mode only); the
    /// end events carry the segment's real status.
    fn emit_segment(
        &self,
        event: StagedEvent,
        segment: &Segment,
        segments: usize,
        slot: usize,
        ok: Option<bool>,
    ) {
        if !m1_trace_enabled() {
            return;
        }
        let blocks = segment.blocks.len() as u64;
        tracing::info!(
            "DYN_M1_TRACE {}",
            serde_json::json!({
                "schema": 1,
                "ts_ns": nvtx::monotonic_ns(),
                "request_id": self.request_id,
                "request_key": self.request_key,
                "request_key_hex": nvtx::key_hex(self.request_key),
                "transfer_id": self.transfer_id,
                "transfer_key": self.transfer_key,
                "transfer_key_hex": nvtx::key_hex(self.transfer_key),
                "component": "physical",
                "event": event.name(),
                "direction": self.direction,
                "tier_code": self.tier_code,
                "blocks": blocks,
                "bytes": blocks.saturating_mul(self.bytes_per_block),
                "status": ok.map(|ok| if ok { "ok" } else { "error" }),
                "segment": segment.index,
                "segments": segments,
                "slot": slot,
            })
        );
    }
}

#[cfg(feature = "nccl")]
use cudarc::nccl::sys::ncclComm_t;

/// Transfer execution mode for distributed workers
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TransferMode {
    /// Each rank manages its own shard independently (default)
    #[default]
    Sharded,
    /// All ranks replicate Device data via NCCL broadcast
    Replicated,
}

/// Thread-safe wrapper for NCCL communicator handle.
///
/// # Safety
/// NCCL communicators are thread-safe once created. All NCCL operations using the same
/// communicator will be serialized internally by NCCL. The raw pointer is safe to send
/// between threads as long as the communicator is not destroyed while in use.
#[cfg(feature = "nccl")]
#[derive(Clone, Copy)]
pub struct NcclCommHandle(ncclComm_t);

#[cfg(feature = "nccl")]
impl NcclCommHandle {
    /// Create a new NcclCommHandle from a raw ncclComm_t.
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - `comm` is a valid NCCL communicator
    /// - The communicator will not be destroyed while this handle exists
    pub unsafe fn new(comm: ncclComm_t) -> Self {
        Self(comm)
    }

    /// Get the raw ncclComm_t handle.
    pub fn as_raw(&self) -> ncclComm_t {
        self.0
    }
}

// Safety: NCCL communicators are thread-safe once created
#[cfg(feature = "nccl")]
unsafe impl Send for NcclCommHandle {}
#[cfg(feature = "nccl")]
unsafe impl Sync for NcclCommHandle {}

/// Inner NCCL configuration (only available with nccl feature)
#[cfg(feature = "nccl")]
#[derive(Clone, Copy)]
struct NcclConfigInner {
    comm: NcclCommHandle,
    rank: i32,
    world_size: i32,
}

/// Transfer mode configuration for replicated transfers.
/// Always available regardless of NCCL feature - use is_enabled() to check.
#[derive(Clone, Copy, Default)]
pub struct NcclConfig {
    #[cfg(feature = "nccl")]
    inner: Option<NcclConfigInner>,
    #[cfg(not(feature = "nccl"))]
    _phantom: (),
}

impl NcclConfig {
    /// Create a disabled/empty config (sharded mode)
    pub fn disabled() -> Self {
        Self::default()
    }

    /// Create an enabled config for replicated mode (only with nccl feature)
    ///
    /// # Preconditions
    /// - `0 <= rank < world_size`
    /// - `world_size > 0`
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - `comm` is a valid NCCL communicator
    /// - The communicator will not be destroyed while this config exists
    #[cfg(feature = "nccl")]
    pub unsafe fn enabled(comm: ncclComm_t, rank: i32, world_size: i32) -> Self {
        unsafe {
            assert!(
                world_size > 0 && (0..world_size).contains(&rank),
                "NCCL topology invariant violated: required 0 <= rank < world_size, world_size > 0; got rank={}, world_size={}",
                rank,
                world_size
            );
            Self {
                inner: Some(NcclConfigInner {
                    comm: NcclCommHandle::new(comm),
                    rank,
                    world_size,
                }),
            }
        }
    }

    /// Returns true if NCCL is enabled and configured
    pub fn is_enabled(&self) -> bool {
        #[cfg(feature = "nccl")]
        {
            self.inner.is_some()
        }
        #[cfg(not(feature = "nccl"))]
        {
            false
        }
    }

    /// Get rank (panics if not enabled)
    pub fn rank(&self) -> i32 {
        #[cfg(feature = "nccl")]
        {
            self.inner.as_ref().expect("NCCL not enabled").rank
        }
        #[cfg(not(feature = "nccl"))]
        {
            panic!("NCCL feature not enabled")
        }
    }

    /// Get world size (panics if not enabled)
    pub fn world_size(&self) -> i32 {
        #[cfg(feature = "nccl")]
        {
            self.inner.as_ref().expect("NCCL not enabled").world_size
        }
        #[cfg(not(feature = "nccl"))]
        {
            panic!("NCCL feature not enabled")
        }
    }

    /// Get the NCCL communicator handle (panics if not enabled)
    #[cfg(feature = "nccl")]
    pub fn comm(&self) -> NcclCommHandle {
        self.inner.as_ref().expect("NCCL not enabled").comm
    }
}

type LocalBlock<S, M> = Block<S, locality::Local, M>;
type LocalBlockDataList<S> = Vec<LocalBlockData<S>>;

/// A batching wrapper for connector transfers to prevent resource exhaustion.
/// Splits large transfers into smaller batches that can be handled by the resource pools.
#[derive(Clone, Debug)]
pub struct ConnectorTransferBatcher {
    max_batch_size: usize,
}

impl ConnectorTransferBatcher {
    pub fn new() -> Self {
        Self {
            max_batch_size: max_transfer_batch_size(),
        }
    }

    pub async fn execute_batched_transfer(
        &self,
        handler: &BlockTransferHandler,
        request: BlockTransferRequest,
    ) -> Result<()> {
        // In replicated mode, execute sequentially (all ranks must participate together)
        // to ensure proper NCCL collective synchronization
        if handler.transfer_mode() == TransferMode::Replicated {
            return handler.execute_transfer_direct(request).await;
        }

        let blocks = request.blocks();
        let num_blocks = blocks.len();

        if num_blocks <= self.max_batch_size {
            return handler.execute_transfer_direct(request).await;
        }

        let batches = blocks.chunks(self.max_batch_size);

        let batch_futures: Vec<_> = batches
            .map(|batch| {
                let batch_request = BlockTransferRequest {
                    from_pool: *request.from_pool(),
                    to_pool: *request.to_pool(),
                    blocks: batch.to_vec(),
                    connector_req: None,
                };
                handler.execute_transfer_direct(batch_request)
            })
            .collect();

        // Execute all batches concurrently
        tracing::debug!("Executing {} batches concurrently", batch_futures.len());

        match try_join_all(batch_futures).await {
            Ok(_) => Ok(()),
            Err(e) => {
                tracing::error!("Batched connector transfer failed: {}", e);
                Err(e)
            }
        }
    }
}

/// A handler for all block transfers. Wraps a group of [`BlockTransferPoolManager`]s.
#[derive(Clone)]
pub struct BlockTransferHandler {
    device: Option<LocalBlockDataList<DeviceStorage>>,
    host: Option<LocalBlockDataList<PinnedStorage>>,
    disk: Option<LocalBlockDataList<DiskStorage>>,
    context: Arc<TransferContext>,
    scheduler_client: Option<TransferSchedulerClient>,
    batcher: ConnectorTransferBatcher,
    /// Pinned bounce buffer for Disk -> Device onboarding without the GDS backend.
    disk_bounce: Option<Arc<DiskBounce>>,
    /// Transfer mode: sharded (default) or replicated
    transfer_mode: TransferMode,
    /// NCCL config (required for replicated mode)
    #[cfg(feature = "nccl")]
    nccl_config: NcclConfig,
}

impl BlockTransferHandler {
    pub fn new(
        device_blocks: Option<Vec<LocalBlock<DeviceStorage, BasicMetadata>>>,
        host_blocks: Option<Vec<LocalBlock<PinnedStorage, BasicMetadata>>>,
        disk_blocks: Option<Vec<LocalBlock<DiskStorage, BasicMetadata>>>,
        context: Arc<TransferContext>,
        scheduler_client: Option<TransferSchedulerClient>,
        nccl_config: NcclConfig,
    ) -> Result<Self> {
        let transfer_mode = if nccl_config.is_enabled() {
            TransferMode::Replicated
        } else {
            TransferMode::Sharded
        };

        Ok(Self {
            device: Self::get_local_data(device_blocks),
            host: Self::get_local_data(host_blocks),
            disk: Self::get_local_data(disk_blocks),
            context,
            scheduler_client,
            batcher: ConnectorTransferBatcher::new(),
            disk_bounce: None,
            transfer_mode,
            #[cfg(feature = "nccl")]
            nccl_config,
        })
    }

    /// Stage Disk -> Device onboarding through `blocks`, a pinned bounce buffer of
    /// `config.slots` x `config.segment_blocks` blocks registered with the NIXL agent.
    pub(crate) fn with_disk_bounce(
        mut self,
        config: DiskBounceConfig,
        blocks: Vec<LocalBlock<PinnedStorage, BasicMetadata>>,
    ) -> Result<Self> {
        let blocks = Self::get_local_data(Some(blocks)).unwrap_or_default();
        self.disk_bounce = Some(Arc::new(DiskBounce::new(config, blocks)?));
        Ok(self)
    }

    /// Returns the transfer mode (sharded or replicated)
    pub fn transfer_mode(&self) -> TransferMode {
        self.transfer_mode
    }

    fn get_local_data<S: Storage>(
        blocks: Option<Vec<LocalBlock<S, BasicMetadata>>>,
    ) -> Option<LocalBlockDataList<S>> {
        blocks.map(|blocks| {
            blocks
                .into_iter()
                .map(|b| {
                    let block_data = b.block_data() as &dyn Any;

                    block_data
                        .downcast_ref::<LocalBlockData<S>>()
                        .unwrap()
                        .clone()
                })
                .collect()
        })
    }

    /// Initiate a transfer between two pools.
    async fn begin_transfer<Source, Target>(
        &self,
        source_pool_list: &Option<LocalBlockDataList<Source>>,
        target_pool_list: &Option<LocalBlockDataList<Target>>,
        request: BlockTransferRequest,
    ) -> Result<tokio::sync::oneshot::Receiver<()>>
    where
        Source: Storage + NixlDescriptor,
        Target: Storage + NixlDescriptor,
        // Check that the source block is readable, local, and writable to the target block.
        LocalBlockData<Source>:
            ReadableBlock<StorageType = Source> + Local + WriteToStrategy<LocalBlockData<Target>>,
        // Check that the target block is writable.
        LocalBlockData<Target>: WritableBlock<StorageType = Target>,
        LocalBlockData<Source>: BlockDataProvider<Locality = locality::Local>,
        LocalBlockData<Target>: BlockDataProviderMut<Locality = locality::Local>,
    {
        let Some(source_pool_list) = source_pool_list else {
            return Err(anyhow::anyhow!("Source pool manager not initialized"));
        };
        let Some(target_pool_list) = target_pool_list else {
            return Err(anyhow::anyhow!("Target pool manager not initialized"));
        };

        // Extract the `from` and `to` indices from the request.
        let source_idxs = request.blocks().iter().map(|(from, _)| *from);
        let target_idxs = request.blocks().iter().map(|(_, to)| *to);

        // Get the blocks corresponding to the indices.
        let sources: Vec<LocalBlockData<Source>> = source_idxs
            .map(|idx| source_pool_list[idx].clone())
            .collect();
        let mut targets: Vec<LocalBlockData<Target>> = target_idxs
            .map(|idx| target_pool_list[idx].clone())
            .collect();

        // Perform the transfer, and return the notifying channel.
        match sources.write_to(&mut targets, self.context.clone()) {
            Ok(channel) => Ok(channel),
            Err(e) => {
                tracing::error!("Failed to write to blocks: {:?}", e);
                Err(e.into())
            }
        }
    }

    /// Execute transfer with batching to prevent resource exhaustion
    pub async fn execute_transfer(&self, request: BlockTransferRequest) -> Result<()> {
        self.execute_transfer_traced(request, None).await
    }

    async fn execute_transfer_traced(
        &self,
        request: BlockTransferRequest,
        trace: Option<&TransferTrace>,
    ) -> Result<()> {
        // Staged onboarding segments the transfer itself, so it bypasses the batcher. In
        // replicated mode it runs on rank 0 inside the replicated path (before the broadcast).
        if self.transfer_mode == TransferMode::Sharded
            && let Some(bounce) = self.staged_bounce(&request)
        {
            return self
                .execute_staged_disk_to_device(bounce, &request, trace)
                .await;
        }
        self.batcher.execute_batched_transfer(self, request).await
    }

    /// The bounce buffer, when `request` is a Disk -> Device transfer that must be staged.
    fn staged_bounce(&self, request: &BlockTransferRequest) -> Option<&Arc<DiskBounce>> {
        match (request.from_pool(), request.to_pool()) {
            (Disk, Device) => self.disk_bounce.as_ref(),
            _ => None,
        }
    }

    /// Disk -> Device through the pinned bounce buffer (see [`super::staged`]). Returns after
    /// every started segment has finished, so a failure leaves no read or copy in flight.
    async fn execute_staged_disk_to_device(
        &self,
        bounce: &Arc<DiskBounce>,
        request: &BlockTransferRequest,
        trace: Option<&TransferTrace>,
    ) -> Result<()> {
        let segments = plan_segments(request.blocks(), bounce.config().segment_blocks);
        tracing::debug!(
            "staged disk onboarding of {} blocks in {} segments",
            request.blocks().len(),
            segments.len()
        );
        let io = StagedSegmentIo {
            handler: self,
            bounce: bounce.as_ref(),
        };
        run_staged(
            &io,
            bounce.slots(),
            &segments,
            |event, segment, slot, ok| {
                if let Some(trace) = trace {
                    trace.emit_segment(event, segment, segments.len(), slot, ok);
                }
            },
        )
        .await
    }

    /// Execute transfer directly without batching (used by the batcher)
    pub async fn execute_transfer_direct(&self, request: BlockTransferRequest) -> Result<()> {
        match self.transfer_mode {
            TransferMode::Sharded => self.execute_transfer_spmd_sharded(request).await,
            #[cfg(feature = "nccl")]
            TransferMode::Replicated => self.execute_transfer_spmd_replicated(request).await,
            #[cfg(not(feature = "nccl"))]
            TransferMode::Replicated => {
                Err(anyhow::anyhow!("Replicated mode requires NCCL feature"))
            }
        }
    }

    /// Execute transfer using sharded mode (each rank manages its own shard independently)
    async fn execute_transfer_spmd_sharded(&self, request: BlockTransferRequest) -> Result<()> {
        tracing::debug!(
            "Performing sharded transfer of {} blocks from {:?} to {:?}",
            request.blocks().len(),
            request.from_pool(),
            request.to_pool()
        );

        tracing::debug!("request: {request:#?}");

        let notify = match (request.from_pool(), request.to_pool()) {
            (Device, Host) => self.begin_transfer(&self.device, &self.host, request).await,
            (Device, Disk) => self.begin_transfer(&self.device, &self.disk, request).await,
            (Host, Device) => self.begin_transfer(&self.host, &self.device, request).await,
            (Host, Disk) => self.begin_transfer(&self.host, &self.disk, request).await,
            (Disk, Device) => self.begin_transfer(&self.disk, &self.device, request).await,
            _ => {
                return Err(anyhow::anyhow!("Invalid transfer type."));
            }
        }?;

        notify.await?;
        Ok(())
    }

    /// Execute transfer using replicated mode (NCCL broadcast for Device blocks)
    #[cfg(feature = "nccl")]
    async fn execute_transfer_spmd_replicated(&self, request: BlockTransferRequest) -> Result<()> {
        assert!(
            self.nccl_config.is_enabled(),
            "NCCL config required for replicated mode"
        );
        let rank = self.nccl_config.rank();
        let is_rank0 = rank == 0;
        let use_bcast = request.to_pool() == &Device && request.from_pool() != &Device;

        if use_bcast {
            tracing::info!(
                "NCCL replicated transfer: {} blocks from {:?} to {:?}, rank={}, \
                 rank0 will load from storage then broadcast to all GPUs",
                request.blocks().len(),
                request.from_pool(),
                request.to_pool(),
                rank
            );
        } else {
            tracing::debug!(
                "Replicated transfer: {} blocks from {:?} to {:?} (rank={}, bcast={})",
                request.blocks().len(),
                request.from_pool(),
                request.to_pool(),
                rank,
                use_bcast
            );
        }

        // Device → Device: all ranks do local transfer (no broadcast)
        if request.from_pool() == &Device && request.to_pool() == &Device {
            return self.execute_transfer_spmd_sharded(request).await;
        }

        // Non-rank0 with no broadcast needed: no-op
        if !is_rank0 && !use_bcast {
            return Ok(());
        }

        // Rank 0 does the actual copy
        if is_rank0 && let Some(bounce) = self.staged_bounce(&request) {
            self.execute_staged_disk_to_device(bounce, &request, None)
                .await?;
        } else if is_rank0 {
            let notify = match (request.from_pool(), request.to_pool()) {
                (Device, Host) => {
                    self.begin_transfer(&self.device, &self.host, request.clone())
                        .await
                }
                (Device, Disk) => {
                    self.begin_transfer(&self.device, &self.disk, request.clone())
                        .await
                }
                (Host, Device) => {
                    self.begin_transfer(&self.host, &self.device, request.clone())
                        .await
                }
                (Host, Disk) => {
                    self.begin_transfer(&self.host, &self.disk, request.clone())
                        .await
                }
                (Disk, Device) => {
                    self.begin_transfer(&self.disk, &self.device, request.clone())
                        .await
                }
                _ => {
                    return Err(anyhow::anyhow!("Invalid transfer type."));
                }
            }?;
            notify.await?;
        }

        // Broadcast Device blocks if needed (all ranks participate)
        if use_bcast {
            self.broadcast_device_blocks(&request).await?;
        }

        Ok(())
    }

    /// Broadcast Device blocks to all ranks using NCCL
    #[cfg(feature = "nccl")]
    async fn broadcast_device_blocks(&self, request: &BlockTransferRequest) -> Result<()> {
        use crate::block_manager::block::transfer::{NcclGroup, bcast_block};

        let device_blocks = self
            .device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Device blocks required for broadcast"))?;

        // Get raw CUstream from the CudaStream wrapper
        let stream = self.context.stream().cu_stream();
        let comm = self.nccl_config.comm();

        // Get destination block indices (the Device blocks to broadcast)
        let dst_indices: Vec<usize> = request.blocks().iter().map(|(_, to)| *to).collect();

        let rank = self.nccl_config.rank();
        let world_size = self.nccl_config.world_size();
        tracing::info!(
            "NCCL broadcast starting: rank={}/{}, num_blocks={}, block_indices={:?}",
            rank,
            world_size,
            dst_indices.len(),
            dst_indices
        );

        // Create NCCL group and broadcast all blocks
        let group = unsafe { NcclGroup::new()? };

        for &block_idx in &dst_indices {
            let block = &device_blocks[block_idx];
            unsafe {
                bcast_block(block, 0, comm.as_raw(), stream)?;
            }
        }

        group.end()?; // Submit the group so we can observe ncclGroupEnd errors
        drop(group);

        // Synchronize: wait for all NCCL operations to complete on the stream
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.context.cuda_event(tx)?;
        rx.await
            .map_err(|_| anyhow::anyhow!("CUDA event channel closed"))?;

        tracing::info!(
            "NCCL broadcast completed: rank={}/{}, num_blocks={}",
            rank,
            world_size,
            dst_indices.len()
        );

        Ok(())
    }
}

/// The two hops of a staged segment on this worker's pools.
struct StagedSegmentIo<'a> {
    handler: &'a BlockTransferHandler,
    bounce: &'a DiskBounce,
}

fn pool_blocks<S: Storage>(
    pool: &Option<LocalBlockDataList<S>>,
    name: &str,
    indices: impl Iterator<Item = usize>,
) -> Result<LocalBlockDataList<S>> {
    let pool = pool
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("{name} pool not initialized"))?;
    indices
        .map(|idx| {
            pool.get(idx)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("{name} block {idx} out of range"))
        })
        .collect()
}

#[async_trait]
impl SegmentIo for StagedSegmentIo<'_> {
    async fn read_into_slot(&self, slot: usize, segment: &Segment) -> Result<()> {
        let sources = pool_blocks(
            &self.handler.disk,
            "disk",
            segment.blocks.iter().map(|(src, _)| *src),
        )?;
        let mut targets = self.bounce.slot_blocks(slot, segment.blocks.len())?;
        read_disk_blocks_into_host(&sources, &mut targets, &self.handler.context)?
            .await
            .map_err(|_| anyhow::anyhow!("NIXL disk read ended without a result"))??;
        if self.bounce.take_injected_read_failure() {
            anyhow::bail!("injected disk read failure ({DISK_BOUNCE_FAIL_READS_ENV})");
        }
        Ok(())
    }

    async fn copy_from_slot(&self, slot: usize, segment: &Segment) -> Result<()> {
        let sources = self.bounce.slot_blocks(slot, segment.blocks.len())?;
        let mut targets = pool_blocks(
            &self.handler.device,
            "device",
            segment.blocks.iter().map(|(_, dst)| *dst),
        )?;
        sources
            .write_to(&mut targets, self.handler.context.clone())?
            .await
            .map_err(|_| anyhow::anyhow!("host -> device copy ended without completing"))?;
        Ok(())
    }
}

#[async_trait]
impl Handler for BlockTransferHandler {
    async fn handle(&self, mut message: MessageHandle) -> Result<()> {
        if message.data.len() != 1 {
            return Err(anyhow::anyhow!(
                "Block transfer request must have exactly one data element"
            ));
        }

        let mut request: BlockTransferRequest = serde_json::from_slice(&message.data[0])?;

        let result = if let Some(req) = request.connector_req.take() {
            let operation_id = req.uuid;
            let canonical_request_id = nvtx::canonical_uuid(&req.request_id);
            let trace_request_id = canonical_request_id
                .clone()
                .unwrap_or_else(|| req.request_id.clone());
            let is_onboard = *request.to_pool() == Device
                && matches!(*request.from_pool(), Host | Disk);
            let tier_code = match *request.from_pool() {
                Host => nvtx::TIER_HOST,
                Disk => nvtx::TIER_DISK,
                Device => nvtx::TIER_GPU,
            };
            let direction = match (*request.from_pool(), *request.to_pool()) {
                (Host, Device) => "h2d",
                (Disk, Device) => "d2d",
                _ => "other",
            };
            let request_key = canonical_request_id
                .as_deref()
                .and_then(nvtx::request_key)
                .unwrap_or(0);
            let transfer_id = operation_id.to_string();
            let transfer_key = nvtx::transfer_key(Some(&transfer_id)).unwrap_or(0);
            let blocks = request.blocks().len() as u64;
            let bytes = blocks.saturating_mul(PHASE_C_BLOCK_BYTES);
            let payload = PhaseCPayload::new(
                request_key,
                transfer_key,
                0,
                tier_code,
                blocks,
                bytes,
            );

            tracing::debug!(
                request_id = %req.request_id,
                operation_id = %operation_id,
                "scheduling transfer"
            );

            let client = self
                .scheduler_client
                .as_ref()
                .expect("scheduler client is required")
                .clone();

            let queue_wait_range = is_onboard.then(|| {
                PhaseCRange::start("transfer_queue_wait", CATEGORY_TRANSFER, payload)
            });
            if is_onboard {
                emit_transfer_boundary(
                    "transfer_enqueued",
                    &trace_request_id,
                    request_key,
                    &transfer_id,
                    transfer_key,
                    direction,
                    tier_code,
                    blocks,
                    bytes,
                    None,
                );
            }
            let handle = client.schedule_transfer(req).await?;
            drop(queue_wait_range);
            if is_onboard {
                emit_transfer_boundary(
                    "transfer_dequeued",
                    &trace_request_id,
                    request_key,
                    &transfer_id,
                    transfer_key,
                    direction,
                    tier_code,
                    blocks,
                    bytes,
                    None,
                );
            }

            // we don't support cancellation yet
            assert_eq!(handle.scheduler_decision(), SchedulingDecision::Execute);

            let dma_range = is_onboard.then(|| {
                PhaseCRange::start(
                    if direction == "h2d" {
                        "h2d_transfer"
                    } else {
                        "d2d_transfer"
                    },
                    CATEGORY_TRANSFER,
                    payload,
                )
            });
            // Keep a disk_read range aligned with the d2d range so the source tier remains
            // directly queryable in Nsight Systems. With GDS the read is fused with the copy;
            // when staged through the bounce buffer this range spans both hops, and the
            // per-segment M1 trace events separate the disk read from the host -> device copy.
            let disk_read_range = (is_onboard && *request.from_pool() == Disk).then(|| {
                PhaseCRange::start("disk_read", CATEGORY_TRANSFER, payload)
            });
            if is_onboard {
                emit_transfer_boundary(
                    "dma_start",
                    &trace_request_id,
                    request_key,
                    &transfer_id,
                    transfer_key,
                    direction,
                    tier_code,
                    blocks,
                    bytes,
                    None,
                );
            }
            // Device blocks that receive KV data; reported to vLLM if the onboarding fails.
            let onboard_block_ids: Vec<usize> = if is_onboard {
                request.blocks().iter().map(|(_, dst)| *dst).collect()
            } else {
                Vec::new()
            };
            let segment_trace = (is_onboard && m1_trace_enabled()).then(|| TransferTrace {
                request_id: trace_request_id.clone(),
                request_key,
                transfer_id: transfer_id.clone(),
                transfer_key,
                direction,
                tier_code,
                bytes_per_block: self
                    .disk_bounce
                    .as_ref()
                    .map_or(PHASE_C_BLOCK_BYTES, |bounce| {
                        bounce.bytes_per_block() as u64
                    }),
            });
            let transfer_result = self
                .execute_transfer_traced(request, segment_trace.as_ref())
                .await;
            drop(disk_read_range);
            drop(dma_range);
            if is_onboard {
                emit_transfer_boundary(
                    "dma_end",
                    &trace_request_id,
                    request_key,
                    &transfer_id,
                    transfer_key,
                    direction,
                    tier_code,
                    blocks,
                    bytes,
                    Some(if transfer_result.is_ok() { "ok" } else { "error" }),
                );
            }

            match transfer_result {
                Ok(_) => {
                    handle.mark_complete(Ok(())).await;
                    Ok(())
                }
                Err(e) => {
                    let completion_error = if is_onboard {
                        anyhow::Error::new(TransferBlocksFailed {
                            block_ids: onboard_block_ids,
                            reason: format!("{e:#}"),
                        })
                    } else {
                        anyhow::anyhow!("{e:#}")
                    };
                    handle.mark_complete(Err(completion_error)).await;
                    Err(e)
                }
            }
        } else {
            self.execute_transfer(request).await
        };

        finish_transfer_message(&mut message, result).await
    }
}

/// Answer a transfer message with the transfer's result rather than a bare ack, so the leader
/// can tell a failed transfer from a completed one. A failed transfer is logged, not returned:
/// an error from a message handler cancels the whole ZMQ worker loop (critical task), after
/// which every later transfer of the worker would hang.
async fn finish_transfer_message(message: &mut MessageHandle, result: Result<()>) -> Result<()> {
    let outcome = TransferOutcome::from_result(&result);
    message
        .reply(ZMQ_TRANSFER_BLOCKS_MESSAGE, &[outcome.encode()])
        .await?;
    if let Err(error) = result {
        tracing::error!("block transfer failed; reported to the leader: {error:#}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::{SinkExt, StreamExt};
    use std::collections::{HashMap, HashSet, VecDeque};
    use std::time::Duration;
    use tmq::{AsZmqSocket, Context, Message, Multipart, publish, pull};
    use tokio_util::sync::CancellationToken;

    /// Stands in for a transfer handler whose transfers all fail.
    struct FailingTransfers;

    #[async_trait]
    impl Handler for FailingTransfers {
        async fn handle(&self, mut message: MessageHandle) -> Result<()> {
            let result = Err(anyhow::anyhow!("disk read failed"));
            finish_transfer_message(&mut message, result).await
        }
    }

    #[tokio::test]
    async fn failed_transfers_are_reported_and_the_worker_keeps_serving() {
        let context = Context::new();
        let mut publisher = publish(&context).bind("tcp://127.0.0.1:*").unwrap();
        let pub_url = publisher.get_socket().get_last_endpoint().unwrap().unwrap();
        let mut replies = pull(&context).bind("tcp://127.0.0.1:*").unwrap();
        let ack_url = replies.get_socket().get_last_endpoint().unwrap().unwrap();

        let mut handlers: HashMap<String, Arc<dyn Handler>> = HashMap::new();
        handlers.insert(
            ZMQ_TRANSFER_BLOCKS_MESSAGE.to_string(),
            Arc::new(FailingTransfers),
        );
        let cancel = CancellationToken::new();
        let _worker =
            ZmqActiveMessageWorker::new(&pub_url, &ack_url, handlers, cancel.clone()).unwrap();

        // Publish until two different messages were answered (early messages can be lost while
        // the subscriber connects). With a dying handler loop only the first would be answered.
        let mut answered = HashSet::new();
        let mut next_id = 0usize;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
        while answered.len() < 2 {
            assert!(
                tokio::time::Instant::now() < deadline,
                "worker stopped answering after {} failed transfers",
                answered.len()
            );
            next_id += 1;
            let mut frames: VecDeque<Message> = VecDeque::new();
            frames.push_back(next_id.to_be_bytes().as_slice().into());
            frames.push_back(ZMQ_TRANSFER_BLOCKS_MESSAGE.into());
            frames.push_back(b"{}".as_slice().into());
            publisher.send(Multipart(frames)).await.unwrap();

            if let Ok(Some(Ok(reply))) =
                tokio::time::timeout(Duration::from_millis(100), replies.next()).await
            {
                assert_eq!(reply.len(), 3, "a transfer reply carries its result");
                let id = usize::from_be_bytes((*reply[0]).try_into().unwrap());
                let outcome: TransferOutcome = serde_json::from_slice(&reply[2]).unwrap();
                assert!(!outcome.ok);
                assert!(outcome.error.unwrap().contains("disk read failed"));
                answered.insert(id);
            }
        }
        assert!(!cancel.is_cancelled());
        cancel.cancel();
    }
}
