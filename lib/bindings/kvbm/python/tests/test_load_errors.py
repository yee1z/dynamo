# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KVBM reports failed KV loads to vLLM through get_block_ids_with_load_errors."""

from types import SimpleNamespace

from kvbm.vllm_integration.connector.dynamo_connector import DynamoConnector
from kvbm.vllm_integration.connector_worker import KvConnectorWorker
from kvbm.vllm_integration.rust import KvConnectorWorker as RustKvConnectorWorker
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1


class _FakeRustWorker:
    def __init__(self, failed):
        self._failed = set(failed)

    def get_block_ids_with_load_errors(self):
        failed, self._failed = self._failed, set()
        return failed


def _worker(failed):
    worker = object.__new__(KvConnectorWorker)
    worker._connector = _FakeRustWorker(failed)
    return worker


def test_worker_reports_failed_blocks_once():
    worker = _worker({3, 7})
    assert worker.get_block_ids_with_load_errors() == {3, 7}
    assert worker.get_block_ids_with_load_errors() == set()


def test_dynamo_connector_delegates_to_its_worker():
    connector = object.__new__(DynamoConnector)
    connector._worker = _worker({5})
    assert connector.get_block_ids_with_load_errors() == {5}
    assert connector.get_block_ids_with_load_errors() == set()


def test_dynamo_connector_overrides_the_vllm_default():
    # The base class always returns an empty set, which hid every failed load.
    assert (
        DynamoConnector.get_block_ids_with_load_errors
        is not KVConnectorBase_V1.get_block_ids_with_load_errors
    )


def test_rust_worker_exposes_load_errors():
    assert RustKvConnectorWorker is not None
    assert callable(
        getattr(RustKvConnectorWorker, "get_block_ids_with_load_errors", None)
    )


def test_scheduler_role_has_no_load_errors_to_report():
    connector = object.__new__(DynamoConnector)
    connector._worker = None
    connector._scheduler = SimpleNamespace()
    assert connector.get_block_ids_with_load_errors() == set()
