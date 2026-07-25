# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned-vLLM worker hook for request-aligned model execution ranges."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time

from vllm.v1.worker.gpu_worker import Worker

from dynamo.common import phase_c_nvtx


logger = logging.getLogger(__name__)


class PhaseCWorker(Worker):
    """Wrap Worker.execute_model's built-in profile context per request.

    Prefill ranges cover the model-runner call for that context batch. Decode
    ranges remain open from the first decode model call until vLLM reports the
    request finished in a later SchedulerOutput.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._phase_c_decode_ranges: dict[str, tuple[object, str]] = {}

    def _emit(self, event: str, request_id: str, engine_request_id: str) -> None:
        if "DYN_M1_TRACE" not in os.environ:
            return
        item = {
            "schema": 1,
            "ts_ns": time.monotonic_ns(),
            "request_id": request_id,
            "engine_request_id": engine_request_id,
            "component": "vllm_model_runner",
            "event": event,
            "worker_id": str(getattr(self, "rank", "")),
            **phase_c_nvtx.trace_keys(request_id),
        }
        logger.info("DYN_M1_TRACE %s", json.dumps(item, separators=(",", ":")))

    @staticmethod
    def _scheduled_requests(scheduler_output):
        for request in scheduler_output.scheduled_new_reqs:
            yield request.req_id, "prefill"
        cached = scheduler_output.scheduled_cached_reqs
        for request_id in cached.req_ids:
            phase = "prefill" if cached.is_context_phase(request_id) else "decode"
            yield request_id, phase

    def _close_finished_decodes(self, scheduler_output) -> None:
        for engine_request_id in scheduler_output.finished_req_ids:
            entry = self._phase_c_decode_ranges.pop(engine_request_id, None)
            if entry is None:
                continue
            range_id, request_id = entry
            phase_c_nvtx.end_range(range_id)
            self._emit("decode_model_end", request_id, engine_request_id)

    @contextlib.contextmanager
    def annotate_profile(self, scheduler_output):
        self._close_finished_decodes(scheduler_output)
        prefill_ranges: list[tuple[object, str, str]] = []
        worker_id = str(getattr(self, "rank", ""))

        for engine_request_id, phase in self._scheduled_requests(scheduler_output):
            try:
                request_id = phase_c_nvtx.canonical_uuid(engine_request_id)
                payload = phase_c_nvtx.make_payload(request_id, worker_id=worker_id)
            except ValueError:
                continue
            if phase == "prefill":
                range_id = phase_c_nvtx.start_range("prefill", payload)
                prefill_ranges.append((range_id, request_id, engine_request_id))
                self._emit("prefill_model_start", request_id, engine_request_id)
            elif engine_request_id not in self._phase_c_decode_ranges:
                range_id = phase_c_nvtx.start_range("decode", payload)
                self._phase_c_decode_ranges[engine_request_id] = (range_id, request_id)
                self._emit("decode_model_start", request_id, engine_request_id)

        try:
            with super().annotate_profile(scheduler_output):
                yield
        finally:
            for range_id, request_id, engine_request_id in reversed(prefill_ranges):
                phase_c_nvtx.end_range(range_id)
                self._emit("prefill_model_end", request_id, engine_request_id)

    def shutdown(self) -> None:
        for engine_request_id, (range_id, request_id) in list(
            self._phase_c_decode_ranges.items()
        ):
            phase_c_nvtx.end_range(range_id)
            self._emit("decode_model_end", request_id, engine_request_id)
        self._phase_c_decode_ranges.clear()
        super().shutdown()
