# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.common import phase_c_nvtx


REQUEST_ID = "369a1572-4253-4632-bfc6-39631d9c98e9"


def test_schema_v1_key_vector() -> None:
    assert phase_c_nvtx.request_key(REQUEST_ID) == 1_893_824_137_375_840_644
    assert phase_c_nvtx.key_hex(phase_c_nvtx.request_key(REQUEST_ID)) == "1a483704df58f984"
    assert phase_c_nvtx.request_key(f"chatcmpl-{REQUEST_ID.upper()}") == 1_893_824_137_375_840_644


def test_payload_is_fixed_u64_vector() -> None:
    payload = phase_c_nvtx.make_payload(
        REQUEST_ID,
        worker_id="worker-0",
        tier_code=phase_c_nvtx.TIER_HOST,
        blocks=3,
    )
    values = payload.values()
    assert len(values) == 7
    assert values[0] == 1
    assert values[1] == 1_893_824_137_375_840_644
    assert values[4:] == (phase_c_nvtx.TIER_HOST, 3, 3 * phase_c_nvtx.BLOCK_BYTES)


def test_payload_rejects_bad_byte_count() -> None:
    with pytest.raises(ValueError, match="does not equal"):
        phase_c_nvtx.Payload(
            request_key=phase_c_nvtx.request_key(REQUEST_ID), blocks=2, bytes=1
        ).values()


def test_only_fixed_messages_are_accepted() -> None:
    payload = phase_c_nvtx.make_payload(REQUEST_ID)
    with pytest.raises(ValueError, match="not a Phase C range"):
        phase_c_nvtx.start_range(f"prefill:{REQUEST_ID}", payload)
