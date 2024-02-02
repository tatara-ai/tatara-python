from typing import Optional
import time
import contextvars
from tatara.tatara_logging.utils import _gen_id_from_trace_and_event

from tatara.tatara_types import LogType
from tatara.tatara_logging._record_keys import (
    LOG_FORMAT_VERSION,
    LOG_RECORD_KEY_HAS_RATING,
    LOG_RECORD_KEY_ID,
    LOG_RECORD_KEY_PROJECT,
    LOG_RECORD_KEY_RATING,
    LOG_RECORD_KEY_TIMESTAMP,
    LOG_RECORD_KEY_TYPE,
    LOG_RECORD_KEY_VERSION,
)
from tatara.tatara_logging.trace import Trace
from tatara.tatara_logging.span import Span
from tatara.tatara_logging.rating import Rating
from tatara.tatara_logging._background_queue_logger import BackgroundLazyQueueLogger
from tatara.network._tatara_network_client import TataraNetworkClient


class TataraClientState:
    current_trace: contextvars.ContextVar[Optional[Trace]]
    current_span: contextvars.ContextVar[Optional[Span]]

    def __init__(
        self,
        project: str,
        queue_size: int,
        flush_interval: float,
        api_key: Optional[str],
        is_dev: bool = False,
    ):
        self.project = project
        self.api_key = api_key
        self.is_dev = is_dev
        self.current_trace = contextvars.ContextVar("current_trace", default=None)
        self.current_span = contextvars.ContextVar("current_span", default=None)

        self.tatara_network_client = TataraNetworkClient(
            api_key=self.api_key, is_dev=is_dev
        )
        self.bglq_logger = BackgroundLazyQueueLogger(
            queue_size, flush_interval=flush_interval,
            tatara_network_client=self.tatara_network_client, api_key=api_key,
        )

    def log_rating(
        self, rating: Rating, trace_id: str, span_event: Optional[str] = None
    ):
        id = _gen_id_from_trace_and_event(trace_id, span_event)
        now = time.time()

        self.bglq_logger.log(
            {
                LOG_RECORD_KEY_PROJECT: self.project,
                LOG_RECORD_KEY_TYPE: LogType.TRACE
                if span_event is None
                else LogType.SPAN,
                LOG_RECORD_KEY_ID: id,
                LOG_RECORD_KEY_TIMESTAMP: now,
                LOG_RECORD_KEY_VERSION: LOG_FORMAT_VERSION,
                LOG_RECORD_KEY_RATING: rating,
                LOG_RECORD_KEY_HAS_RATING: 1,
            }
        )

        if span_event is not None:
            # mark that the span's parent trace has a rating
            self.bglq_logger.log(
                {
                    LOG_RECORD_KEY_PROJECT: self.project,
                    LOG_RECORD_KEY_TYPE: LogType.TRACE,
                    LOG_RECORD_KEY_ID: trace_id,
                    LOG_RECORD_KEY_TIMESTAMP: now,
                    LOG_RECORD_KEY_VERSION: LOG_FORMAT_VERSION,
                    LOG_RECORD_KEY_HAS_RATING: 1,
                }
            )
