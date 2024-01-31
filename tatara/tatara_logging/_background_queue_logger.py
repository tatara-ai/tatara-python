import atexit
import base64
import json
import queue
import threading
import traceback
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Dict, List, Optional

from tatara.network._tatara_network_client import TataraNetworkClient
from ._record_keys import (
    LOG_RECORD_KEY_ID,
    LOG_RECORD_KEY_METADATA,
    LOG_RECORD_KEY_PROPERTIES,
    LOG_RECORD_PROPERTIES_KEY_DIFFUSION_EVENT,
    LOG_RECORD_PROPERTIES_KEY_END_TIME,
    LOG_RECORD_PROPERTIES_KEY_LLM_EVENT,
    LOG_RECORD_PROPERTIES_KEY_START_TIME,
)


def _custom_serializer(obj):
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, bytes):
        return base64.b64encode(obj).decode()
    elif is_dataclass(obj):
        return asdict(obj)
    else:
        return "not serializable"


def _merge_dicts_on_key(output_dict: Dict, d1: Dict, d2: Dict, key: str):
    if key in output_dict:
        output_dict[key] = {**d1[key], **d2[key]}
    else:
        output_dict[key] = d2[key]


# TODO: benchmark this and see if it's actually more performant than a generic recurisve merge
def _merge_logs(log1: Dict, log2: Dict):
    merged_log = dict(log1)
    for key in log2:
        if key == LOG_RECORD_KEY_METADATA:
            _merge_dicts_on_key(merged_log, log1, log2, LOG_RECORD_KEY_METADATA)
        elif key == LOG_RECORD_KEY_PROPERTIES:
            log1_properties = log1[key]
            log2_properties = log2[key]
            if key in merged_log:
                if LOG_RECORD_PROPERTIES_KEY_START_TIME in log2_properties:
                    merged_log[key][
                        LOG_RECORD_PROPERTIES_KEY_START_TIME
                    ] = log2_properties[LOG_RECORD_PROPERTIES_KEY_START_TIME]
                if LOG_RECORD_PROPERTIES_KEY_END_TIME in log2_properties:
                    merged_log[key][
                        LOG_RECORD_PROPERTIES_KEY_END_TIME
                    ] = log2_properties[LOG_RECORD_PROPERTIES_KEY_END_TIME]
                if LOG_RECORD_PROPERTIES_KEY_LLM_EVENT in log2_properties:
                    _merge_dicts_on_key(
                        merged_log[key],
                        log1_properties,
                        log2_properties,
                        LOG_RECORD_PROPERTIES_KEY_LLM_EVENT,
                    )
                if LOG_RECORD_PROPERTIES_KEY_DIFFUSION_EVENT in log2_properties:
                    _merge_dicts_on_key(
                        merged_log[key],
                        log1_properties,
                        log2_properties,
                        LOG_RECORD_PROPERTIES_KEY_DIFFUSION_EVENT,
                    )

            else:
                merged_log[key] = log2[key]
        else:
            merged_log[key] = log2[key]

    return merged_log


# TODO: possibly more efficient to have an indexable queue structure that gets updated on writes
def _merge_logs_and_convert_to_json(logs: List[Dict]) -> List[str]:
    merged_logs = dict()
    for log in logs:
        log_id = log[LOG_RECORD_KEY_ID]
        if log_id in merged_logs:
            merged_logs[log_id] = _merge_logs(merged_logs[log_id], log)

        else:
            merged_logs[log_id] = log

    json_list = [
        json.dumps(log, default=_custom_serializer)
        for log in list(merged_logs.values())
    ]
    return json_list[::-1]


# queue logs and periodically send over the wire
class BackgroundLazyQueueLogger:
    def __init__(
        self, queue_size: int, flush_interval: float, tatara_network_client: TataraNetworkClient,api_key: Optional[str] = None
    ):
        self._flush_lock = threading.RLock()
        self._start_thread_lock = threading.RLock()
        self._log_queue = queue.Queue(queue_size)
        self._flush_interval = flush_interval
        self._thread = None
        self._timer = None
        self._queue_full = threading.Semaphore(value=0)
        self._tatara_network_client = tatara_network_client

        atexit.register(
            lambda: (print("Flushing remaining Tatara logs..."), self._flush())
        )

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _start(self) -> None:
        if not self.is_alive:
            with self._start_thread_lock:
                if not self.is_alive:
                    self._thread = threading.Thread(
                        target=self._target,
                        name="tatara.BackgroundLazyQueueLogger",
                        daemon=True,
                    )
                    self._thread.start()

                    self._timer = threading.Timer(
                        interval=self._flush_interval, function=self._flush
                    )
                    self._timer.daemon = True
                    self._timer.start()

    def _target(self):
        while True:
            self._queue_full.acquire()
            try:
                self._flush()
            except Exception:
                traceback.print_exc()

    def _flush(self):
        with self._flush_lock:
            log_items = []
            try:
                for _ in range(self._log_queue.qsize()):
                    log_items.append(self._log_queue.get_nowait())
            except queue.Empty:
                pass

            if len(log_items) == 0:
                return

            json_logs_to_flush = _merge_logs_and_convert_to_json(log_items)

            max_batch_size = 100
            max_request_size = 3 * 1024 * 1024
            while True:
                json_log_str = "["
                batch_size = 0
                while (
                    batch_size < max_batch_size and len(json_log_str) < max_request_size
                ):
                    if len(json_logs_to_flush) > 0:
                        json_log_str += json_logs_to_flush.pop() + ","
                        batch_size += 1
                    else:
                        break

                if json_log_str == "[":
                    break

                # Replace last comma with a closing bracket for valid JSON
                json_log_str = json_log_str[:-1] + "]"
                self._tatara_network_client.send_logs_post_request(json_log_str)

    def log(self, log_dict: Dict):
        self._start()
        try:
            self._log_queue.put_nowait(log_dict)
        except queue.Full:
            self._queue_full.release()
            self._log_queue.put(log_dict)
