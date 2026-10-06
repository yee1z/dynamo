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


def test_worker_request_received_event_schema() -> None:
    event = phase_c_nvtx.worker_request_received_event(
        f"chatcmpl-{REQUEST_ID}",
        ts_ns=123,
        worker_id=7587,
        dp_rank=0,
        prompt_tokens=5,
        fpm_worker_id="w0",
    )
    assert event == {
        "schema": 1,
        "ts_ns": 123,
        "request_id": REQUEST_ID,
        "handler_request_id": f"chatcmpl-{REQUEST_ID}",
        "component": "vllm_handler",
        "event": "worker_request_received",
        "worker_id": 7587,
        "dp_rank": 0,
        "fpm_worker_id": "w0",
        "prompt_tokens": 5,
        "clock": "CLOCK_MONOTONIC",
        "request_key": 1_893_824_137_375_840_644,
        "request_key_hex": "1a483704df58f984",
        "transfer_key": 0,
        "transfer_key_hex": "0000000000000000",
    }


def test_worker_request_received_requires_canonical_uuid() -> None:
    assert (
        phase_c_nvtx.worker_request_received_event(
            "req-1", ts_ns=1, worker_id=None, dp_rank=None, prompt_tokens=None
        )
        is None
    )


def test_prompt_token_count() -> None:
    assert phase_c_nvtx.prompt_token_count({"token_ids": [1, 2, 3]}) == 3
    assert phase_c_nvtx.prompt_token_count({"messages": []}) is None
    assert phase_c_nvtx.prompt_token_count({"token_ids": "abc"}) is None
    assert phase_c_nvtx.prompt_token_count(None) is None
