// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NVTX timeline-annotation helpers for Nsight Systems profiling.
//!
//! Delegates to [`cudarc::nvtx`] for the actual NVTX calls
//!
//! # Gating (two-level)
//!
//! | Cargo feature `nvtx` | `DYN_ENABLE_RUST_NVTX` env | Effect                                    |
//! |----------------------|----------------------------|-------------------------------------------|
//! | off (default)        | any                        | macros compile to nothing; zero overhead  |
//! | on                   | unset                      | one `Relaxed` load per site (~1 ns)       |
//! | on                   | `1` / `true` / `yes`       | cudarc NVTX calls (~50 ns/annotation)     |
//!
//! # Usage
//!
//! ```rust,ignore
//! let _r = dynamo_nvtx_range!("preprocess.tokenize"); // RAII — pops at scope end
//! dynamo_nvtx_push!("codec.encode");
//! dynamo_nvtx_pop!();
//! dynamo_nvtx_name_thread!("tokio-worker-0");
//! ```
//!
//! # Build
//!
//! ```bash
//! cargo build --profile profiling --features nvtx
//! ```
//! Requires `libnvToolsExt.so` at runtime (CUDA Toolkit or NVHPC).

use sha2::{Digest, Sha256};

pub const PHASE_C_SCHEMA_VERSION: u64 = 1;
pub const PHASE_C_DOMAIN: &str = "dynamo.phase_c";
pub const PHASE_C_BLOCK_BYTES: u64 = 2_359_296;

pub const CATEGORY_ROUTER: u32 = 1;
pub const CATEGORY_SCHEDULER: u32 = 2;
pub const CATEGORY_CONNECTOR: u32 = 3;
pub const CATEGORY_TRANSFER: u32 = 4;
pub const CATEGORY_MODEL: u32 = 5;

pub const TIER_UNKNOWN: u64 = 0;
pub const TIER_MISS: u64 = 1;
pub const TIER_GPU: u64 = 2;
pub const TIER_HOST: u64 = 3;
pub const TIER_DISK: u64 = 4;

/// Return a lowercase canonical UUID embedded in a protocol request ID.
pub fn canonical_uuid(value: &str) -> Option<String> {
    if let Ok(id) = uuid::Uuid::parse_str(value) {
        return Some(id.to_string());
    }
    if value.len() < 36 {
        return None;
    }
    (0..=value.len() - 36).find_map(|offset| {
        value
            .get(offset..offset + 36)
            .and_then(|candidate| uuid::Uuid::parse_str(candidate).ok())
            .map(|id| id.to_string())
    })
}

pub fn stable_key(value: &str) -> u64 {
    let digest = Sha256::digest(value.as_bytes());
    u64::from_be_bytes(
        digest[..8]
            .try_into()
            .expect("SHA-256 prefix is eight bytes"),
    )
}

pub fn request_key(request_id: &str) -> Option<u64> {
    canonical_uuid(request_id).map(|id| stable_key(&id))
}

pub fn transfer_key(transfer_id: Option<&str>) -> Option<u64> {
    match transfer_id {
        None => Some(0),
        Some(id) => request_key(id),
    }
}

pub fn key_hex(key: u64) -> String {
    format!("{key:016x}")
}

/// CLOCK_MONOTONIC timestamp shared by the cooperating Dynamo processes.
pub fn monotonic_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` points to writable storage and CLOCK_MONOTONIC takes no
    // additional arguments.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    debug_assert_eq!(rc, 0);
    (ts.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(ts.tv_nsec as u64)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct PhaseCPayload(pub [u64; 7]);

impl PhaseCPayload {
    pub fn new(
        request_key: u64,
        transfer_key: u64,
        worker_key: u64,
        tier_code: u64,
        blocks: u64,
        bytes: u64,
    ) -> Self {
        debug_assert!(bytes == 0 || bytes == blocks.saturating_mul(PHASE_C_BLOCK_BYTES));
        Self([
            PHASE_C_SCHEMA_VERSION,
            request_key,
            transfer_key,
            worker_key,
            tier_code,
            blocks,
            bytes,
        ])
    }

    pub fn for_request(request_id: &str) -> Option<Self> {
        Some(Self::new(
            request_key(request_id)?, 0, 0, TIER_UNKNOWN, 0, 0,
        ))
    }
}

#[cfg(feature = "nvtx")]
mod phase_c_impl {
    use super::*;
    use cudarc::nvtx::sys;
    use std::ffi::CStr;
    use std::sync::OnceLock;

    const NVTX_VERSION: u16 = 2;
    const NVTX_MESSAGE_TYPE_REGISTERED: i32 = 3;
    const NVTX_PAYLOAD_TYPE_EXT: i32 = 0xDFBD0009_u32 as i32;
    const NVTX_TYPE_PAYLOAD_SCHEMA_RAW: u64 = 1023;

    #[repr(C)]
    struct ExtendedPayloadData {
        schema_id: u64,
        size: usize,
        payload: *const core::ffi::c_void,
    }

    struct State {
        domain: usize,
        messages: [usize; 10],
    }

    unsafe impl Send for State {}
    unsafe impl Sync for State {}

    static STATE: OnceLock<Option<State>> = OnceLock::new();

    fn env_enabled() -> bool {
        std::env::var("DYN_PHASE_C_NVTX")
            .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false)
    }

    fn cstr(bytes: &'static [u8]) -> &'static CStr {
        CStr::from_bytes_with_nul(bytes).expect("static NVTX string is NUL terminated")
    }

    fn message_index(message: &str) -> Option<usize> {
        match message {
            "router_match" => Some(0),
            "worker_queue" => Some(1),
            "connector_match" => Some(2),
            "onboard_submit" => Some(3),
            "transfer_queue_wait" => Some(4),
            "disk_read" => Some(5),
            "h2d_transfer" => Some(6),
            "d2d_transfer" => Some(7),
            "prefill" => Some(8),
            "decode" => Some(9),
            _ => None,
        }
    }

    fn message_cstr(index: usize) -> &'static CStr {
        cstr(match index {
            0 => b"router_match\0",
            1 => b"worker_queue\0",
            2 => b"connector_match\0",
            3 => b"onboard_submit\0",
            4 => b"transfer_queue_wait\0",
            5 => b"disk_read\0",
            6 => b"h2d_transfer\0",
            7 => b"d2d_transfer\0",
            8 => b"prefill\0",
            9 => b"decode\0",
            _ => unreachable!(),
        })
    }

    fn state() -> Option<&'static State> {
        STATE.get_or_init(|| {
            if !env_enabled() {
                return None;
            }
            unsafe {
                let domain = sys::nvtxDomainCreateA(cstr(b"dynamo.phase_c\0").as_ptr());
                if domain.is_null() {
                    return None;
                }
                for (category, name) in [
                    (CATEGORY_ROUTER, b"router\0".as_slice()),
                    (CATEGORY_SCHEDULER, b"scheduler\0".as_slice()),
                    (CATEGORY_CONNECTOR, b"connector\0".as_slice()),
                    (CATEGORY_TRANSFER, b"transfer\0".as_slice()),
                    (CATEGORY_MODEL, b"model\0".as_slice()),
                ] {
                    sys::nvtxDomainNameCategoryA(domain, category, cstr(name).as_ptr());
                }
                let mut messages = [0usize; 10];
                for (index, slot) in messages.iter_mut().enumerate() {
                    *slot = sys::nvtxDomainRegisterStringA(
                        domain,
                        message_cstr(index).as_ptr(),
                    ) as usize;
                }
                Some(State {
                    domain: domain as usize,
                    messages,
                })
            }
        })
        .as_ref()
    }

    unsafe fn attributes(
        state: &State,
        message: &str,
        category: u32,
        extended: &ExtendedPayloadData,
    ) -> sys::nvtxEventAttributes_t {
        let message_handle = state.messages[message_index(message).expect("fixed Phase C message")]
            as sys::nvtxStringHandle_t;
        sys::nvtxEventAttributes_t {
            version: NVTX_VERSION,
            size: std::mem::size_of::<sys::nvtxEventAttributes_t>() as u16,
            category,
            colorType: 0,
            color: 0,
            payloadType: NVTX_PAYLOAD_TYPE_EXT,
            reserved0: 1,
            payload: sys::nvtxEventAttributes_v2_payload_t {
                ullValue: extended as *const ExtendedPayloadData as u64,
            },
            messageType: NVTX_MESSAGE_TYPE_REGISTERED,
            message: sys::nvtxMessageValue_t { registered: message_handle },
        }
    }

    pub(super) fn start(
        message: &'static str,
        category: u32,
        payload: PhaseCPayload,
    ) -> PhaseCRange {
        let Some(state) = state() else {
            return PhaseCRange { inner: None };
        };
        assert!(message_index(message).is_some(), "non-contract Phase C message");
        let extended = ExtendedPayloadData {
            schema_id: NVTX_TYPE_PAYLOAD_SCHEMA_RAW,
            size: std::mem::size_of_val(&payload.0),
            payload: payload.0.as_ptr().cast(),
        };
        let id = unsafe {
            let attrs = attributes(state, message, category, &extended);
            sys::nvtxDomainRangeStartEx(state.domain as sys::nvtxDomainHandle_t, &attrs)
        };
        PhaseCRange { inner: Some((state.domain, id)) }
    }

    pub(super) fn mark(message: &'static str, category: u32, payload: PhaseCPayload) {
        let Some(state) = state() else {
            return;
        };
        assert!(message_index(message).is_some(), "non-contract Phase C message");
        let extended = ExtendedPayloadData {
            schema_id: NVTX_TYPE_PAYLOAD_SCHEMA_RAW,
            size: std::mem::size_of_val(&payload.0),
            payload: payload.0.as_ptr().cast(),
        };
        unsafe {
            let attrs = attributes(state, message, category, &extended);
            sys::nvtxDomainMarkEx(state.domain as sys::nvtxDomainHandle_t, &attrs);
        }
    }

    pub(super) fn end(inner: (usize, u64)) {
        unsafe {
            sys::nvtxDomainRangeEnd(inner.0 as sys::nvtxDomainHandle_t, inner.1)
        }
    }
}

pub struct PhaseCRange {
    #[cfg(feature = "nvtx")]
    inner: Option<(usize, u64)>,
}

impl PhaseCRange {
    pub fn start(message: &'static str, category: u32, payload: PhaseCPayload) -> Self {
        #[cfg(feature = "nvtx")]
        {
            return phase_c_impl::start(message, category, payload);
        }
        #[cfg(not(feature = "nvtx"))]
        {
            let _ = (message, category, payload);
            Self {}
        }
    }
}

impl Drop for PhaseCRange {
    fn drop(&mut self) {
        #[cfg(feature = "nvtx")]
        if let Some(inner) = self.inner.take() {
            phase_c_impl::end(inner);
        }
    }
}

pub fn phase_c_mark(message: &'static str, category: u32, payload: PhaseCPayload) {
    #[cfg(feature = "nvtx")]
    phase_c_impl::mark(message, category, payload);
    #[cfg(not(feature = "nvtx"))]
    let _ = (message, category, payload);
}

#[cfg(test)]
mod phase_c_tests {
    use super::*;
    const REQUEST_ID: &str = "369a1572-4253-4632-bfc6-39631d9c98e9";

    #[test]
    fn schema_v1_key_matches_python_vector() {
        assert_eq!(request_key(REQUEST_ID), Some(1_893_824_137_375_840_644));
        assert_eq!(
            request_key(&format!("chatcmpl-{}", REQUEST_ID.to_uppercase())),
            Some(1_893_824_137_375_840_644)
        );
        assert_eq!(key_hex(request_key(REQUEST_ID).unwrap()), "1a483704df58f984");
    }

    #[test]
    fn payload_has_fixed_seven_u64_layout() {
        let payload = PhaseCPayload::new(1, 2, 3, TIER_DISK, 5, 5 * PHASE_C_BLOCK_BYTES);
        assert_eq!(
            payload.0,
            [1, 1, 2, 3, 4, 5, 5 * PHASE_C_BLOCK_BYTES]
        );
        assert_eq!(
            std::mem::size_of::<PhaseCPayload>(),
            7 * std::mem::size_of::<u64>()
        );
    }

    #[test]
    fn range_and_mark_api_smoke() {
        let payload = PhaseCPayload::for_request(REQUEST_ID).unwrap();
        let range = PhaseCRange::start("prefill", CATEGORY_MODEL, payload);
        phase_c_mark("onboard_submit", CATEGORY_CONNECTOR, payload);
        drop(range);
    }
}

#[cfg(feature = "nvtx")]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "nvtx")]
static NVTX_ENABLED: AtomicBool = AtomicBool::new(false);

// ── Public API ───────────────────────────────────────────────────────────────

/// Initialise the NVTX subsystem from the `DYN_ENABLE_RUST_NVTX` environment variable.
/// Must be called once at runtime startup before any annotation macros fire.
/// No-op when the `nvtx` Cargo feature is off.
pub fn init() {
    #[cfg(feature = "nvtx")]
    {
        let enabled = std::env::var("DYN_ENABLE_RUST_NVTX")
            .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);
        NVTX_ENABLED.store(enabled, Ordering::Relaxed);
        if enabled {
            tracing::info!("NVTX annotations enabled (DYN_ENABLE_RUST_NVTX)");
        }
    }
}

/// Returns `true` when the `nvtx` feature is compiled in **and** `DYN_ENABLE_RUST_NVTX` is set.
#[inline(always)]
pub fn enabled() -> bool {
    #[cfg(feature = "nvtx")]
    {
        return NVTX_ENABLED.load(Ordering::Relaxed);
    }
    #[allow(unreachable_code)]
    false
}

/// Push an NVTX range onto the calling thread's stack.
/// No-op (compiled out) when the `nvtx` feature is off.
#[inline(always)]
pub fn push_impl(name: &str) {
    #[cfg(feature = "nvtx")]
    {
        if NVTX_ENABLED.load(Ordering::Relaxed) {
            cudarc::nvtx::result::range_push(name);
        }
    }
    let _ = name;
}

/// Pop the innermost NVTX range from the calling thread's stack.
/// No-op (compiled out) when the `nvtx` feature is off.
#[inline(always)]
pub fn pop_impl() {
    #[cfg(feature = "nvtx")]
    {
        if NVTX_ENABLED.load(Ordering::Relaxed) {
            cudarc::nvtx::result::range_pop();
        }
    }
}

/// Name the current OS thread in the Nsight Systems timeline.
/// No-op (compiled out) when the `nvtx` feature is off.
#[inline(always)]
pub fn name_current_thread_impl(name: &str) {
    #[cfg(feature = "nvtx")]
    {
        if NVTX_ENABLED.load(Ordering::Relaxed) {
            #[cfg(target_os = "linux")]
            let tid = unsafe { libc::syscall(libc::SYS_gettid) as u32 };
            #[cfg(not(target_os = "linux"))]
            let tid = 0u32;
            cudarc::nvtx::result::name_os_thread(tid, name);
        }
    }
    let _ = name;
}

// ── RAII guard ───────────────────────────────────────────────────────────────

/// RAII guard that pops an NVTX range when dropped.
/// Construct with [`dynamo_nvtx_range!`].
#[cfg(feature = "nvtx")]
pub struct NvtxRangeGuard {
    active: bool,
}

/// Zero-sized no-op guard used when the `nvtx` feature is off.
#[cfg(not(feature = "nvtx"))]
pub struct NvtxRangeGuard;

impl NvtxRangeGuard {
    #[doc(hidden)]
    pub fn new(name: &str) -> Self {
        #[cfg(feature = "nvtx")]
        {
            let active = NVTX_ENABLED.load(Ordering::Relaxed);
            if active {
                cudarc::nvtx::result::range_push(name);
            }
            return NvtxRangeGuard { active };
        }
        #[cfg(not(feature = "nvtx"))]
        {
            let _ = name;
            NvtxRangeGuard {}
        }
    }
}

#[cfg(feature = "nvtx")]
impl Drop for NvtxRangeGuard {
    fn drop(&mut self) {
        if self.active {
            cudarc::nvtx::result::range_pop();
        }
    }
}

#[cfg(not(feature = "nvtx"))]
impl Drop for NvtxRangeGuard {
    fn drop(&mut self) {}
}

// ── Macros ───────────────────────────────────────────────────────────────────

/// Push a named NVTX range onto the calling thread's stack.
/// Zero-cost when the `nvtx` Cargo feature is off.
#[macro_export]
macro_rules! dynamo_nvtx_push {
    ($name:expr) => {
        $crate::nvtx::push_impl($name)
    };
}

/// Pop the innermost NVTX range from the calling thread's stack.
/// Zero-cost when the `nvtx` Cargo feature is off.
#[macro_export]
macro_rules! dynamo_nvtx_pop {
    () => {
        $crate::nvtx::pop_impl()
    };
}

/// Open a named NVTX range that closes automatically at end of scope.
///
/// ```rust,ignore
/// let _r = dynamo_nvtx_range!("preprocess.tokenize");
/// // range closes here
/// ```
/// Zero-cost when the `nvtx` Cargo feature is off.
#[macro_export]
macro_rules! dynamo_nvtx_range {
    ($name:expr) => {
        $crate::nvtx::NvtxRangeGuard::new($name)
    };
}

/// Annotate the current OS thread in the Nsight Systems timeline.
/// Zero-cost when the `nvtx` Cargo feature is off.
#[macro_export]
macro_rules! dynamo_nvtx_name_thread {
    ($name:expr) => {
        $crate::nvtx::name_current_thread_impl($name)
    };
}
