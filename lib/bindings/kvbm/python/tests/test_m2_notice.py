# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from kvbm.vllm_integration.connector_leader import (
    KvConnectorLeader,
    _global_dp_rank,
    _numeric_worker_id,
    _served_model,
)


class _FakeConnector:
    def __init__(self):
        self.events = []

    def observe_speculative_onboarding_notice(self, *args):
        self.events.append(("observe", args))
        return "accepted"

    def has_slot(self, request_id):
        self.events.append(("has_slot", request_id))
        return True

    def get_num_new_matched_tokens(self, request_id, request_num_tokens, computed):
        self.events.append(("match", request_id, request_num_tokens, computed))
        return 32, True


def _leader(connector):
    leader = object.__new__(KvConnectorLeader)
    leader._connector = connector
    leader.engine_id = "random-engine-id"
    leader._m2_worker_id = 7
    leader._m2_dp_rank = 0
    leader._m2_model = "model"
    return leader


def _request(extra_args):
    return SimpleNamespace(
        request_id="request-1",
        num_tokens=64,
        sampling_params=SimpleNamespace(extra_args=extra_args),
        lora_request=None,
        cache_salt=None,
    )


def test_worker_and_model_identity_resolution(monkeypatch):
    monkeypatch.setenv("DYN_FPM_WORKER_ID", "17")
    assert _numeric_worker_id("random-engine-id") == 17
    monkeypatch.delenv("DYN_FPM_WORKER_ID")
    assert _numeric_worker_id("namespace.component.backend.9") == 9
    assert _numeric_worker_id("random-engine-id") is None

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_index=None,
            data_parallel_rank=None,
        ),
        model_config=SimpleNamespace(served_model_name=["served"], model="model"),
    )
    assert _global_dp_rank(config) == 0
    assert _served_model(config) == "served"


def test_notice_is_observed_before_slot_lookup_and_actual_match():
    connector = _FakeConnector()
    leader = _leader(connector)
    notice = {"schema": 1, "request_id": "request-1"}

    result = leader.get_num_new_matched_tokens(
        _request({"dynamo": {"m2_speculative_onboarding_notice": notice}}),
        0,
    )

    assert result == (32, True)
    assert [event[0] for event in connector.events] == ["observe", "has_slot", "match"]


def test_feature_off_path_does_not_call_observer():
    connector = _FakeConnector()
    leader = _leader(connector)

    result = leader.get_num_new_matched_tokens(_request(None), 0)

    assert result == (32, True)
    assert [event[0] for event in connector.events] == ["has_slot", "match"]
