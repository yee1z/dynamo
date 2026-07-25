# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-aligned Phase C NVTX schema v1 helpers."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Final


SCHEMA_VERSION: Final = 1
DOMAIN_NAME: Final = "dynamo.phase_c"
BLOCK_BYTES: Final = 2_359_296

CATEGORY_ROUTER: Final = 1
CATEGORY_SCHEDULER: Final = 2
CATEGORY_CONNECTOR: Final = 3
CATEGORY_TRANSFER: Final = 4
CATEGORY_MODEL: Final = 5

TIER_UNKNOWN: Final = 0
TIER_MISS: Final = 1
TIER_GPU: Final = 2
TIER_HOST: Final = 3
TIER_DISK: Final = 4

RANGE_CATEGORIES: Final = {
    "router_match": CATEGORY_ROUTER,
    "worker_queue": CATEGORY_SCHEDULER,
    "connector_match": CATEGORY_CONNECTOR,
    "transfer_queue_wait": CATEGORY_TRANSFER,
    "disk_read": CATEGORY_TRANSFER,
    "h2d_transfer": CATEGORY_TRANSFER,
    "d2d_transfer": CATEGORY_TRANSFER,
    "prefill": CATEGORY_MODEL,
    "decode": CATEGORY_MODEL,
}
MARK_CATEGORIES: Final = {"onboard_submit": CATEGORY_CONNECTOR}

_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_ENABLED = os.getenv("DYN_PHASE_C_NVTX", "").lower() in {"1", "true", "yes", "on"}

if _ENABLED:
    import numpy as _np
    import nvtx as _nvtx

    _DOMAIN = _nvtx.get_domain(DOMAIN_NAME)
else:
    _DOMAIN = None


def canonical_uuid(value: str) -> str:
    """Return a lowercase canonical UUID embedded in *value*, or raise."""
    match = _UUID_RE.search(str(value))
    if not match:
        raise ValueError(f"no canonical UUID in {value!r}")
    return match.group(0).lower()


def stable_key(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def request_key(request_id: str) -> int:
    return stable_key(canonical_uuid(request_id))


def transfer_key(transfer_id: str | None) -> int:
    return 0 if not transfer_id else stable_key(canonical_uuid(transfer_id))


def worker_key(worker_id: str | int | None) -> int:
    if worker_id is None or str(worker_id) == "":
        return 0
    return stable_key(str(worker_id).lower())


def key_hex(key: int) -> str:
    if not 0 <= key <= (1 << 64) - 1:
        raise ValueError(f"key does not fit u64: {key}")
    return f"{key:016x}"


@dataclass(frozen=True, slots=True)
class Payload:
    request_key: int
    transfer_key: int = 0
    worker_key: int = 0
    tier_code: int = TIER_UNKNOWN
    blocks: int = 0
    bytes: int = 0

    def values(self) -> tuple[int, int, int, int, int, int, int]:
        values = (
            SCHEMA_VERSION,
            self.request_key,
            self.transfer_key,
            self.worker_key,
            self.tier_code,
            self.blocks,
            self.bytes,
        )
        if any(not isinstance(value, int) or not 0 <= value < (1 << 64) for value in values):
            raise ValueError(f"Phase C payload values must be u64: {values!r}")
        if self.bytes and self.bytes != self.blocks * BLOCK_BYTES:
            raise ValueError(
                f"bytes={self.bytes} does not equal blocks={self.blocks} * {BLOCK_BYTES}"
            )
        return values


def make_payload(
    request_id: str,
    *,
    transfer_id: str | None = None,
    worker_id: str | int | None = None,
    tier_code: int = TIER_UNKNOWN,
    blocks: int = 0,
    bytes_: int | None = None,
) -> Payload:
    return Payload(
        request_key=request_key(request_id),
        transfer_key=transfer_key(transfer_id),
        worker_key=worker_key(worker_id),
        tier_code=tier_code,
        blocks=blocks,
        bytes=blocks * BLOCK_BYTES if bytes_ is None else bytes_,
    )


def trace_keys(request_id: str, transfer_id: str | None = None) -> dict[str, int | str]:
    req_key = request_key(request_id)
    xfer_key = transfer_key(transfer_id)
    return {
        "request_key": req_key,
        "request_key_hex": key_hex(req_key),
        "transfer_key": xfer_key,
        "transfer_key_hex": key_hex(xfer_key),
    }


def start_range(message: str, payload: Payload):
    category = RANGE_CATEGORIES.get(message)
    if category is None:
        raise ValueError(f"not a Phase C range message: {message!r}")
    values = payload.values()
    if _DOMAIN is None:
        return None
    attributes = _DOMAIN.get_event_attributes(
        message=message,
        category=category,
        payload=_np.asarray(values, dtype=_np.uint64),
    )
    return _DOMAIN.start_range(attributes)


def end_range(range_id) -> None:
    if _DOMAIN is not None and range_id is not None:
        _DOMAIN.end_range(range_id)


def mark(message: str, payload: Payload) -> None:
    category = MARK_CATEGORIES.get(message)
    if category is None:
        raise ValueError(f"not a Phase C mark message: {message!r}")
    values = payload.values()
    if _DOMAIN is not None:
        _DOMAIN.mark(
            message=message,
            category=category,
            payload=_np.asarray(values, dtype=_np.uint64),
        )
