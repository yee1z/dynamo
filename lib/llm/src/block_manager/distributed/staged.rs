// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-staged disk onboarding (G3 -> G1) for workers without the GDS backend.
//!
//! With `DYN_KVBM_NIXL_BACKEND_GDS=false` the NIXL agent only has the POSIX backend, which
//! cannot address device memory, so a direct Disk -> Device transfer fails. The worker then
//! onboards disk blocks in two hops through a dedicated pinned "bounce" buffer:
//!
//! ```text
//! disk blocks --(NIXL POSIX, O_DIRECT)--> bounce slot --(host -> device CUDA copy)--> device blocks
//! ```
//!
//! A transfer is split into segments of at most `segment_blocks` blocks. A segment holds one
//! bounce slot from the start of its disk read until its copy has finished, so with two or more
//! slots the read of one segment overlaps the copy of another. The slots are shared by all
//! concurrent transfers of the worker and handed out in FIFO order; the time spent waiting for a
//! slot is part of the transfer. The bounce buffer is not part of the host (G2) pool, so G2
//! contents and eviction are unaffected.
//!
//! The newer `kvbm-physical` engine plans the same route (`TransferPlan::TwoHop` through pinned
//! memory) when direct device <-> disk transfers are unavailable.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use anyhow::{Result, anyhow, bail};
use async_trait::async_trait;
use tokio::sync::{Semaphore, SemaphorePermit};

use crate::block_manager::block::{BlockDataExt, data::local::LocalBlockData};
use crate::block_manager::storage::PinnedStorage;

/// Blocks per segment, which is also the size of one bounce slot.
pub const DISK_BOUNCE_BLOCKS_ENV: &str = "DYN_KVBM_DISK_BOUNCE_BLOCKS";
/// Number of bounce slots (2 = double buffering).
pub const DISK_BOUNCE_SLOTS_ENV: &str = "DYN_KVBM_DISK_BOUNCE_SLOTS";
/// Fault injection for validation runs only: the first N staged disk-read segments report an
/// error after their read has completed. Never set in formal runs.
pub const DISK_BOUNCE_FAIL_READS_ENV: &str = "DYN_KVBM_DISK_BOUNCE_FAIL_READS";

pub const DEFAULT_DISK_BOUNCE_BLOCKS: usize = 256;
pub const DEFAULT_DISK_BOUNCE_SLOTS: usize = 2;

/// Address and length alignment that O_DIRECT reads into the bounce buffer require.
pub const DIRECT_IO_ALIGNMENT: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiskBounceConfig {
    pub segment_blocks: usize,
    pub slots: usize,
    pub inject_read_failures: usize,
}

impl DiskBounceConfig {
    pub fn from_env() -> Result<Self> {
        Self::parse(
            env_value(DISK_BOUNCE_BLOCKS_ENV)?.as_deref(),
            env_value(DISK_BOUNCE_SLOTS_ENV)?.as_deref(),
            env_value(DISK_BOUNCE_FAIL_READS_ENV)?.as_deref(),
        )
    }

    fn parse(blocks: Option<&str>, slots: Option<&str>, fail_reads: Option<&str>) -> Result<Self> {
        let config = Self {
            segment_blocks: parse_count(
                DISK_BOUNCE_BLOCKS_ENV,
                blocks,
                DEFAULT_DISK_BOUNCE_BLOCKS,
                1,
            )?,
            slots: parse_count(DISK_BOUNCE_SLOTS_ENV, slots, DEFAULT_DISK_BOUNCE_SLOTS, 1)?,
            inject_read_failures: parse_count(DISK_BOUNCE_FAIL_READS_ENV, fail_reads, 0, 0)?,
        };
        if config.segment_blocks.checked_mul(config.slots).is_none() {
            bail!(
                "{DISK_BOUNCE_BLOCKS_ENV}={} x {DISK_BOUNCE_SLOTS_ENV}={} overflows",
                config.segment_blocks,
                config.slots
            );
        }
        Ok(config)
    }

    /// Number of blocks in the whole bounce buffer.
    pub fn total_blocks(&self) -> usize {
        self.segment_blocks * self.slots
    }
}

fn env_value(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(anyhow!("{name}: {error}")),
    }
}

/// Parse a count setting; an unset variable gives `default`, anything else must be an integer
/// of at least `min` (configuration errors fail worker start-up instead of being ignored).
fn parse_count(name: &str, value: Option<&str>, default: usize, min: usize) -> Result<usize> {
    let Some(value) = value else {
        return Ok(default);
    };
    let parsed: usize = value
        .trim()
        .parse()
        .map_err(|error| anyhow!("{name}={value:?} is not a non-negative integer: {error}"))?;
    if parsed < min {
        bail!("{name}={parsed} must be at least {min}");
    }
    Ok(parsed)
}

/// One part of a staged transfer: up to `segment_blocks` (disk source, device target) pairs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Segment {
    pub index: usize,
    pub blocks: Vec<(usize, usize)>,
}

/// Split the transfer's (disk source, device target) pairs into segments, keeping their order.
pub(crate) fn plan_segments(blocks: &[(usize, usize)], segment_blocks: usize) -> Vec<Segment> {
    blocks
        .chunks(segment_blocks.max(1))
        .enumerate()
        .map(|(index, chunk)| Segment {
            index,
            blocks: chunk.to_vec(),
        })
        .collect()
}

/// Index in the bounce buffer of the `offset`-th block of `slot`.
pub(crate) fn bounce_block_index(slot: usize, offset: usize, segment_blocks: usize) -> usize {
    slot * segment_blocks + offset
}

/// The bounce slots of one worker, handed out in FIFO order.
pub(crate) struct BounceSlots {
    free: Mutex<Vec<usize>>,
    permits: Semaphore,
}

/// A held bounce slot; dropping it returns the slot.
pub(crate) struct SlotGuard<'a> {
    slot: usize,
    free: &'a Mutex<Vec<usize>>,
    _permit: SemaphorePermit<'a>,
}

impl BounceSlots {
    pub fn new(slots: usize) -> Self {
        Self {
            free: Mutex::new((0..slots).rev().collect()),
            permits: Semaphore::new(slots),
        }
    }

    /// Wait for a free slot. Waiters are served in the order in which they started waiting.
    pub async fn acquire(&self) -> Result<SlotGuard<'_>> {
        let permit = self
            .permits
            .acquire()
            .await
            .map_err(|_| anyhow!("bounce slot semaphore closed"))?;
        let slot = self
            .free
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .pop()
            .ok_or_else(|| anyhow!("no free bounce slot although a permit was granted"))?;
        Ok(SlotGuard {
            slot,
            free: &self.free,
            _permit: permit,
        })
    }

    /// Slots not held right now.
    #[cfg(test)]
    pub fn available(&self) -> usize {
        self.permits.available_permits()
    }
}

impl SlotGuard<'_> {
    pub fn slot(&self) -> usize {
        self.slot
    }
}

impl Drop for SlotGuard<'_> {
    fn drop(&mut self) {
        // The slot goes back before the permit (a field, dropped after this body), so a waiter
        // woken by the permit always finds a free slot.
        self.free
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(self.slot);
    }
}

/// Remaining injected read failures (`DYN_KVBM_DISK_BOUNCE_FAIL_READS`).
pub(crate) struct InjectedReadFailures(AtomicUsize);

impl InjectedReadFailures {
    pub fn new(count: usize) -> Self {
        Self(AtomicUsize::new(count))
    }

    /// Consume one injected failure; false once none remain.
    pub fn take(&self) -> bool {
        self.0
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
            .is_ok()
    }

    #[cfg(test)]
    pub fn remaining(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

/// A step of a staged segment, reported to the trace callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StagedEvent {
    DiskReadStart,
    DiskReadEnd,
    BounceCopyStart,
    BounceCopyEnd,
}

impl StagedEvent {
    pub fn name(self) -> &'static str {
        match self {
            StagedEvent::DiskReadStart => "disk_read_start",
            StagedEvent::DiskReadEnd => "disk_read_end",
            StagedEvent::BounceCopyStart => "bounce_copy_start",
            StagedEvent::BounceCopyEnd => "bounce_copy_end",
        }
    }
}

/// The two hops of a segment.
#[async_trait]
pub(crate) trait SegmentIo: Send + Sync {
    /// Read the segment's disk blocks into bounce `slot`.
    async fn read_into_slot(&self, slot: usize, segment: &Segment) -> Result<()>;
    /// Copy bounce `slot` into the segment's device blocks.
    async fn copy_from_slot(&self, slot: usize, segment: &Segment) -> Result<()>;
}

/// Run every segment through the bounce slots.
///
/// Returns only after every segment that started has finished, so no read or copy still uses a
/// bounce slot or a device block afterwards. Once a segment fails, segments that have not started
/// yet are skipped. `trace` receives (event, segment, slot, ok) where `ok` is set on end events.
pub(crate) async fn run_staged<Io, Trace>(
    io: &Io,
    slots: &BounceSlots,
    segments: &[Segment],
    trace: Trace,
) -> Result<()>
where
    Io: SegmentIo + ?Sized,
    Trace: Fn(StagedEvent, &Segment, usize, Option<bool>) + Sync,
{
    let failed = AtomicBool::new(false);
    // join_all (not try_join_all): a failure must not drop segments whose read or copy is still
    // in flight. The futures are polled in order, so segments queue for slots in order.
    let results = futures::future::join_all(
        segments
            .iter()
            .map(|segment| run_segment(io, slots, segment, &failed, &trace)),
    )
    .await;
    let mut errors = results.into_iter().filter_map(Result::err);
    match errors.next() {
        None => Ok(()),
        // Segments start in order and are skipped only after a failure, so the first error in
        // segment order is a real failure rather than a skip.
        Some(first) => {
            let failed_segments = 1 + errors.count();
            Err(first.context(format!(
                "staged disk onboarding failed: {failed_segments} of {} segments did not complete",
                segments.len()
            )))
        }
    }
}

async fn run_segment<Io, Trace>(
    io: &Io,
    slots: &BounceSlots,
    segment: &Segment,
    failed: &AtomicBool,
    trace: &Trace,
) -> Result<()>
where
    Io: SegmentIo + ?Sized,
    Trace: Fn(StagedEvent, &Segment, usize, Option<bool>) + Sync,
{
    let guard = slots.acquire().await?;
    if failed.load(Ordering::Acquire) {
        bail!(
            "segment {} skipped after an earlier segment failed",
            segment.index
        );
    }
    let slot = guard.slot();

    trace(StagedEvent::DiskReadStart, segment, slot, None);
    let read = io.read_into_slot(slot, segment).await;
    trace(StagedEvent::DiskReadEnd, segment, slot, Some(read.is_ok()));
    if let Err(error) = read {
        failed.store(true, Ordering::Release);
        return Err(error.context(format!("disk read of segment {}", segment.index)));
    }

    trace(StagedEvent::BounceCopyStart, segment, slot, None);
    let copy = io.copy_from_slot(slot, segment).await;
    trace(
        StagedEvent::BounceCopyEnd,
        segment,
        slot,
        Some(copy.is_ok()),
    );
    if let Err(error) = copy {
        failed.store(true, Ordering::Release);
        return Err(error.context(format!("bounce copy of segment {}", segment.index)));
    }

    // The slot is released only after the copy out of it has finished.
    drop(guard);
    Ok(())
}

/// The pinned bounce buffer of a worker and its slots.
pub struct DiskBounce {
    config: DiskBounceConfig,
    blocks: Vec<LocalBlockData<PinnedStorage>>,
    bytes_per_block: usize,
    slots: BounceSlots,
    injected_read_failures: InjectedReadFailures,
}

impl DiskBounce {
    pub(crate) fn new(
        config: DiskBounceConfig,
        blocks: Vec<LocalBlockData<PinnedStorage>>,
    ) -> Result<Self> {
        if blocks.len() != config.total_blocks() {
            bail!(
                "bounce buffer has {} blocks; {} slots x {} blocks were configured",
                blocks.len(),
                config.slots,
                config.segment_blocks
            );
        }
        let mut bytes_per_block = 0;
        for (index, block) in blocks.iter().enumerate() {
            let regions = block_regions(block)?;
            if index == 0 {
                bytes_per_block = regions.iter().map(|(_, size)| size).sum();
            }
            if let Some((addr, size)) = first_misaligned(regions, DIRECT_IO_ALIGNMENT) {
                bail!(
                    "bounce block {} region at {addr:#x} ({size} bytes) is not \
                     {DIRECT_IO_ALIGNMENT}-byte aligned; O_DIRECT disk reads into it would fail",
                    block.block_id()
                );
            }
        }
        Ok(Self {
            slots: BounceSlots::new(config.slots),
            injected_read_failures: InjectedReadFailures::new(config.inject_read_failures),
            config,
            blocks,
            bytes_per_block,
        })
    }

    pub fn config(&self) -> DiskBounceConfig {
        self.config
    }

    /// Size of one KV block in bytes (from the bounce layout).
    pub fn bytes_per_block(&self) -> usize {
        self.bytes_per_block
    }

    pub(crate) fn slots(&self) -> &BounceSlots {
        &self.slots
    }

    /// The first `len` blocks of bounce `slot`.
    pub(crate) fn slot_blocks(
        &self,
        slot: usize,
        len: usize,
    ) -> Result<Vec<LocalBlockData<PinnedStorage>>> {
        if slot >= self.config.slots || len > self.config.segment_blocks {
            bail!(
                "bounce slot {slot} with {len} blocks is outside {} slots x {} blocks",
                self.config.slots,
                self.config.segment_blocks
            );
        }
        Ok((0..len)
            .map(|offset| {
                self.blocks[bounce_block_index(slot, offset, self.config.segment_blocks)].clone()
            })
            .collect())
    }

    /// Consume one injected read failure, if any remain.
    pub(crate) fn take_injected_read_failure(&self) -> bool {
        self.injected_read_failures.take()
    }
}

/// The (address, size) memory regions that NIXL transfers of `block` use: the whole block when it
/// is contiguous, one region per layer and outer dimension otherwise.
fn block_regions(block: &LocalBlockData<PinnedStorage>) -> Result<Vec<(usize, usize)>> {
    let mut regions = Vec::new();
    if block.is_fully_contiguous() {
        let view = block.block_view()?;
        // SAFETY: only the address is read; nothing is dereferenced.
        regions.push((unsafe { view.as_ptr() } as usize, view.size()));
    } else {
        for layer in 0..block.num_layers() {
            for outer in 0..block.num_outer_dims() {
                let view = block.layer_view(layer, outer)?;
                // SAFETY: only the address is read; nothing is dereferenced.
                regions.push((unsafe { view.as_ptr() } as usize, view.size()));
            }
        }
    }
    Ok(regions)
}

/// First (address, size) region whose address or size is not a multiple of `alignment`.
pub(crate) fn first_misaligned(
    regions: impl IntoIterator<Item = (usize, usize)>,
    alignment: usize,
) -> Option<(usize, usize)> {
    regions
        .into_iter()
        .find(|(addr, size)| addr % alignment != 0 || size % alignment != 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    fn pairs(n: usize) -> Vec<(usize, usize)> {
        (0..n).map(|i| (100 + i, 1000 + i)).collect()
    }

    #[test]
    fn config_defaults_to_two_slots_of_256_blocks() {
        let config = DiskBounceConfig::parse(None, None, None).unwrap();
        assert_eq!(
            config,
            DiskBounceConfig {
                segment_blocks: 256,
                slots: 2,
                inject_read_failures: 0
            }
        );
        assert_eq!(config.total_blocks(), 512);
    }

    #[test]
    fn config_reads_explicit_values() {
        let config = DiskBounceConfig::parse(Some("64"), Some("3"), Some("1")).unwrap();
        assert_eq!(
            config,
            DiskBounceConfig {
                segment_blocks: 64,
                slots: 3,
                inject_read_failures: 1
            }
        );
    }

    #[test]
    fn config_rejects_zero_and_garbage() {
        assert!(DiskBounceConfig::parse(Some("0"), None, None).is_err());
        assert!(DiskBounceConfig::parse(None, Some("0"), None).is_err());
        assert!(DiskBounceConfig::parse(Some("-1"), None, None).is_err());
        assert!(DiskBounceConfig::parse(None, None, Some("many")).is_err());
        assert!(DiskBounceConfig::parse(Some(&usize::MAX.to_string()), Some("2"), None).is_err());
        // Zero injected failures is the normal setting.
        assert!(DiskBounceConfig::parse(None, None, Some("0")).is_ok());
    }

    #[test]
    fn plan_fewer_blocks_than_a_segment() {
        let segments = plan_segments(&pairs(3), 4);
        assert_eq!(
            segments,
            vec![Segment {
                index: 0,
                blocks: pairs(3)
            }]
        );
    }

    #[test]
    fn plan_exactly_one_segment() {
        let segments = plan_segments(&pairs(4), 4);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].blocks, pairs(4));
    }

    #[test]
    fn plan_more_blocks_than_a_segment_keeps_order_across_segments() {
        let blocks = pairs(10);
        let segments = plan_segments(&blocks, 4);
        assert_eq!(segments.len(), 3);
        assert_eq!(
            segments.iter().map(|s| s.blocks.len()).collect::<Vec<_>>(),
            vec![4, 4, 2]
        );
        assert_eq!(
            segments.iter().map(|s| s.index).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        // Every (source, target) pair appears once, in the original order.
        let flattened: Vec<_> = segments.iter().flat_map(|s| s.blocks.clone()).collect();
        assert_eq!(flattened, blocks);
        assert_eq!(segments[1].blocks[0], (104, 1004));
        assert_eq!(segments[2].blocks[1], (109, 1009));
    }

    #[test]
    fn plan_empty_transfer() {
        assert!(plan_segments(&[], 4).is_empty());
    }

    #[test]
    fn bounce_index_is_slot_major() {
        assert_eq!(bounce_block_index(0, 0, 256), 0);
        assert_eq!(bounce_block_index(0, 255, 256), 255);
        assert_eq!(bounce_block_index(1, 0, 256), 256);
        assert_eq!(bounce_block_index(1, 255, 256), 511);
    }

    #[test]
    fn misaligned_regions_are_found() {
        assert_eq!(
            first_misaligned([(0x1000, 4096), (0x3000, 8192)], 4096),
            None
        );
        assert_eq!(
            first_misaligned([(0x1000, 4096), (0x2010, 4096)], 4096),
            Some((0x2010, 4096))
        );
        assert_eq!(
            first_misaligned([(0x1000, 4000)], 4096),
            Some((0x1000, 4000))
        );
        // A 2,359,296-byte block (576 x 4 KiB) at a page-aligned address is aligned.
        assert_eq!(
            first_misaligned([(0x7f00_0000_0000, 2_359_296)], 4096),
            None
        );
    }

    #[test]
    fn injected_failures_are_consumed_once() {
        let injected = InjectedReadFailures::new(2);
        assert!(injected.take());
        assert!(injected.take());
        assert!(!injected.take());
        assert_eq!(injected.remaining(), 0);
        assert!(!InjectedReadFailures::new(0).take());
    }

    #[tokio::test]
    async fn slots_are_released_and_reused() {
        let slots = BounceSlots::new(2);
        let a = slots.acquire().await.unwrap();
        let b = slots.acquire().await.unwrap();
        assert_ne!(a.slot(), b.slot());
        assert_eq!(slots.available(), 0);
        let released = a.slot();
        drop(a);
        assert_eq!(slots.available(), 1);
        let c = slots.acquire().await.unwrap();
        assert_eq!(c.slot(), released);
        drop(b);
        drop(c);
        assert_eq!(slots.available(), 2);
    }

    #[tokio::test]
    async fn slot_waiters_are_served_in_order() {
        let slots = Arc::new(BounceSlots::new(1));
        let held = slots.acquire().await.unwrap();
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut waiters = Vec::new();
        for id in 0..3 {
            let slots = slots.clone();
            let order = order.clone();
            waiters.push(tokio::spawn(async move {
                let guard = slots.acquire().await.unwrap();
                order.lock().unwrap().push(id);
                drop(guard);
            }));
            // Make sure waiter `id` is queued before the next one starts waiting.
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        drop(held);
        for waiter in waiters {
            waiter.await.unwrap();
        }
        assert_eq!(*order.lock().unwrap(), vec![0, 1, 2]);
    }

    /// Records every call and lets tests delay or fail individual segments.
    #[derive(Default)]
    struct FakeIo {
        log: Mutex<Vec<(String, usize, usize)>>,
        fail_read: Mutex<Option<usize>>,
        fail_copy: Mutex<Option<usize>>,
        read_delay_ms: HashMap<usize, u64>,
        in_flight: AtomicUsize,
        max_in_flight: AtomicUsize,
        slot_users: Mutex<HashMap<usize, usize>>,
    }

    impl FakeIo {
        fn enter(&self, slot: usize, segment: usize) {
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_in_flight.fetch_max(now, Ordering::SeqCst);
            let mut users = self.slot_users.lock().unwrap();
            if let Some(other) = users.insert(slot, segment) {
                assert_eq!(other, segment, "slot {slot} used by two segments at once");
            }
        }

        fn leave(&self, slot: usize) {
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            self.slot_users.lock().unwrap().remove(&slot);
        }

        fn calls(&self, kind: &str) -> Vec<usize> {
            self.log
                .lock()
                .unwrap()
                .iter()
                .filter(|(k, _, _)| k == kind)
                .map(|(_, segment, _)| *segment)
                .collect()
        }
    }

    #[async_trait]
    impl SegmentIo for FakeIo {
        async fn read_into_slot(&self, slot: usize, segment: &Segment) -> Result<()> {
            self.enter(slot, segment.index);
            let delay = self.read_delay_ms.get(&segment.index).copied().unwrap_or(5);
            tokio::time::sleep(Duration::from_millis(delay)).await;
            self.log
                .lock()
                .unwrap()
                .push(("read".to_string(), segment.index, slot));
            if *self.fail_read.lock().unwrap() == Some(segment.index) {
                // The slot stays in use until copy or failure handling is over.
                self.slot_users.lock().unwrap().remove(&slot);
                self.in_flight.fetch_sub(1, Ordering::SeqCst);
                bail!("read failed for segment {}", segment.index);
            }
            Ok(())
        }

        async fn copy_from_slot(&self, slot: usize, segment: &Segment) -> Result<()> {
            tokio::time::sleep(Duration::from_millis(2)).await;
            self.log
                .lock()
                .unwrap()
                .push(("copy".to_string(), segment.index, slot));
            self.leave(slot);
            if *self.fail_copy.lock().unwrap() == Some(segment.index) {
                bail!("copy failed for segment {}", segment.index);
            }
            Ok(())
        }
    }

    type TraceLog = Mutex<Vec<(StagedEvent, usize, usize, Option<bool>)>>;

    fn tracer(log: &TraceLog) -> impl Fn(StagedEvent, &Segment, usize, Option<bool>) + Sync + '_ {
        move |event, segment, slot, ok| log.lock().unwrap().push((event, segment.index, slot, ok))
    }

    #[tokio::test]
    async fn every_segment_is_read_then_copied_with_bounded_slots() {
        let io = FakeIo::default();
        let slots = BounceSlots::new(2);
        let segments = plan_segments(&pairs(10), 2);
        let trace = TraceLog::default();
        run_staged(&io, &slots, &segments, tracer(&trace))
            .await
            .unwrap();

        assert_eq!(io.calls("read").len(), 5);
        let mut copied = io.calls("copy");
        copied.sort();
        assert_eq!(copied, vec![0, 1, 2, 3, 4]);
        // Never more segments in flight than slots, and all slots are free afterwards.
        assert!(io.max_in_flight.load(Ordering::SeqCst) <= 2);
        assert_eq!(slots.available(), 2);

        // Each segment emits read start/end then copy start/end on the same slot.
        let trace = trace.lock().unwrap();
        for segment in 0..5 {
            let events: Vec<_> = trace.iter().filter(|e| e.1 == segment).collect();
            assert_eq!(
                events.iter().map(|e| e.0).collect::<Vec<_>>(),
                vec![
                    StagedEvent::DiskReadStart,
                    StagedEvent::DiskReadEnd,
                    StagedEvent::BounceCopyStart,
                    StagedEvent::BounceCopyEnd
                ]
            );
            assert!(events.iter().all(|e| e.2 == events[0].2));
            assert_eq!(events[1].3, Some(true));
            assert_eq!(events[3].3, Some(true));
        }
    }

    #[tokio::test]
    async fn two_slots_overlap_one_read_with_another_copy() {
        // Segment 1 reads slowly; segment 0's copy must not wait for it.
        let io = FakeIo {
            read_delay_ms: HashMap::from([(0, 5), (1, 60)]),
            ..Default::default()
        };
        let slots = BounceSlots::new(2);
        let segments = plan_segments(&pairs(4), 2);
        run_staged(&io, &slots, &segments, |_, _, _, _| {})
            .await
            .unwrap();
        let log = io.log.lock().unwrap();
        let order: Vec<_> = log.iter().map(|(k, s, _)| format!("{k}{s}")).collect();
        assert_eq!(order, vec!["read0", "copy0", "read1", "copy1"]);
    }

    #[tokio::test]
    async fn one_slot_serializes_segments() {
        let io = FakeIo::default();
        let slots = BounceSlots::new(1);
        let segments = plan_segments(&pairs(6), 2);
        run_staged(&io, &slots, &segments, |_, _, _, _| {})
            .await
            .unwrap();
        assert_eq!(io.max_in_flight.load(Ordering::SeqCst), 1);
        let log = io.log.lock().unwrap();
        let order: Vec<_> = log.iter().map(|(k, s, _)| format!("{k}{s}")).collect();
        assert_eq!(
            order,
            vec!["read0", "copy0", "read1", "copy1", "read2", "copy2"]
        );
    }

    #[tokio::test]
    async fn a_failed_read_fails_the_transfer_and_skips_unstarted_segments() {
        let io = FakeIo {
            fail_read: Mutex::new(Some(1)),
            ..Default::default()
        };
        let slots = BounceSlots::new(2);
        let segments = plan_segments(&pairs(10), 2);
        let trace = TraceLog::default();
        let error = run_staged(&io, &slots, &segments, tracer(&trace))
            .await
            .unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("read failed for segment 1"), "{message}");
        assert!(message.contains("segments did not complete"), "{message}");

        // Segment 1 never copies; segment 0 (already started) still finishes.
        assert!(!io.calls("copy").contains(&1));
        assert!(io.calls("copy").contains(&0));
        // Segments queued behind the failure never touch the disk.
        assert!(io.calls("read").len() < segments.len());
        // All slots are returned even on failure.
        assert_eq!(slots.available(), 2);
        // The failed read is traced with ok=false and no copy event follows it.
        let trace = trace.lock().unwrap();
        let segment_one: Vec<_> = trace
            .iter()
            .filter(|e| e.1 == 1)
            .map(|e| (e.0, e.3))
            .collect();
        assert_eq!(
            segment_one,
            vec![
                (StagedEvent::DiskReadStart, None),
                (StagedEvent::DiskReadEnd, Some(false))
            ]
        );
    }

    #[tokio::test]
    async fn a_failed_copy_fails_the_transfer() {
        let io = FakeIo {
            fail_copy: Mutex::new(Some(0)),
            ..Default::default()
        };
        let slots = BounceSlots::new(2);
        let segments = plan_segments(&pairs(2), 2);
        let error = run_staged(&io, &slots, &segments, |_, _, _, _| {})
            .await
            .unwrap_err();
        assert!(format!("{error:#}").contains("copy failed for segment 0"));
        assert_eq!(slots.available(), 2);
    }

    #[tokio::test]
    async fn concurrent_transfers_share_the_slots() {
        let io = Arc::new(FakeIo::default());
        let slots = Arc::new(BounceSlots::new(2));
        let mut transfers = Vec::new();
        for t in 0..3 {
            let io = io.clone();
            let slots = slots.clone();
            transfers.push(tokio::spawn(async move {
                let blocks: Vec<_> = (0..6).map(|i| (t * 100 + i, t * 100 + i)).collect();
                let segments = plan_segments(&blocks, 2);
                run_staged(io.as_ref(), slots.as_ref(), &segments, |_, _, _, _| {}).await
            }));
        }
        for transfer in transfers {
            transfer.await.unwrap().unwrap();
        }
        assert_eq!(io.calls("copy").len(), 9);
        assert!(io.max_in_flight.load(Ordering::SeqCst) <= 2);
        assert_eq!(slots.available(), 2);
    }

    #[tokio::test]
    async fn empty_transfer_succeeds_without_io() {
        let io = FakeIo::default();
        let slots = BounceSlots::new(2);
        run_staged(&io, &slots, &[], |_, _, _, _| {}).await.unwrap();
        assert!(io.log.lock().unwrap().is_empty());
    }
}
