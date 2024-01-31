from .trace import Trace
from ._background_queue_logger import BackgroundLazyQueueLogger
from tatara.tatara_types import (
    LogType,
)
from .span_impl import _SpanImpl
from .rating import Rating
from .utils import _check_event
from typing import Optional, Dict, Any, List
import uuid
import time

from ._record_keys import (
    LOG_FORMAT_VERSION,
    LOG_RECORD_KEY_EVENT,
    LOG_RECORD_KEY_ID,
    LOG_RECORD_KEY_INTERNAL_ID,
    LOG_RECORD_KEY_PROJECT,
    LOG_RECORD_KEY_PROPERTIES,
    LOG_RECORD_KEY_TIMESTAMP,
    LOG_RECORD_KEY_TYPE,
    LOG_RECORD_KEY_VERSION,
    LOG_RECORD_PROPERTIES_KEY_END_TIME,
    LOG_RECORD_PROPERTIES_KEY_START_TIME,
    LOG_RECORD_KEY_METADATA,
    LOG_RECORD_KEY_SPANS,
    LOG_RECORD_KEY_USER_ID,
)

from tatara.client_state import _get_client_state


class _TraceImpl(Trace):
    def __init__(
        self,
        event,
        logger: BackgroundLazyQueueLogger,
        id_: Optional[str] = None,
        user_id: Optional[str] = None,
        start_time: Optional[float] = None,
    ):
        _check_event(event)
        self._event = event
        self._internal_id = "t_" + str(uuid.uuid4())
        self._id = id_ or self._internal_id
        self._user_id = user_id
        self._logger = logger
        now = time.time()
        self._start_time = start_time or now
        self._end_time = None
        self._span_ids = []
        self._metadata = {}
        self._token = _get_client_state().current_trace.set(self)

        self._properties = {
            LOG_RECORD_PROPERTIES_KEY_START_TIME: self._start_time,
            LOG_RECORD_PROPERTIES_KEY_END_TIME: None,
        }

        self._log_record = {
            LOG_RECORD_KEY_PROJECT: _get_client_state().project,
            LOG_RECORD_KEY_TYPE: LogType.TRACE,
            LOG_RECORD_KEY_TIMESTAMP: now,
            LOG_RECORD_KEY_VERSION: LOG_FORMAT_VERSION,
            LOG_RECORD_KEY_INTERNAL_ID: self._internal_id,
            LOG_RECORD_KEY_ID: self._id,
            LOG_RECORD_KEY_USER_ID: self._user_id,
            LOG_RECORD_KEY_EVENT: self._event,
            LOG_RECORD_KEY_SPANS: self._span_ids,
            LOG_RECORD_KEY_PROPERTIES: self._properties,
            LOG_RECORD_KEY_METADATA: self._metadata,
        }

        self._logger.log(self._log_record)

    @property
    def id(self) -> str:
        return self._id

    @property
    def event(self) -> str:
        return self._event

    @property
    def span_ids(self) -> List[str]:
        return self._span_ids

    def _generate_id_for_span_event_and_log(self, event: str) -> str:
        span_id = event + "_" + self.id

        # add suffix if we're calling the same span event multiple times
        suffix = 2
        while span_id in self.span_ids:
            span_id = event + "_" + str(suffix) + "_" + self.id
            suffix += 1

        self._span_ids.append(span_id)

        self._log_record[LOG_RECORD_KEY_SPANS] = self._span_ids
        self._log()

        return span_id

    def start_span(self, event: str, parent_event: Optional[str]):
        self._check_finished()
        _check_event(event)

        span_id = self._generate_id_for_span_event_and_log(event)
        # TODO: handle edge case if multiple spans with the same event are called
        parent_id = (
            self.id
            if (parent_event is None or parent_event == self.event)
            else parent_event + "_" + self.id
        )
        return _SpanImpl(
            event=event,
            id_=span_id,
            parent_id=parent_id,
            trace=self,
            background_queue_logger=_get_client_state().bglq_logger,
            start_time=time.time(),
        )

    def end(self, end_time: Optional[float] = None):
        self._check_finished()

        self._end_time = end_time or time.time()
        _get_client_state().current_trace.reset(self._token)

        self._log_record[LOG_RECORD_KEY_PROPERTIES][
            LOG_RECORD_PROPERTIES_KEY_END_TIME
        ] = self._end_time
        self._log()

    def log_metadata(self, metadata: Dict[str, Any]):
        self._check_finished()

        self._metadata.update(metadata)
        self._log_record[LOG_RECORD_KEY_METADATA] = self._metadata
        self._log()

    # we can log a rating after a trace is finished
    def log_rating(self, rating: Rating):
        return _get_client_state().log_rating(
            rating=rating, trace_id=self.id, span_event=None
        )

    def _check_finished(self):
        if self._end_time:
            raise RuntimeError("Cannot call method on a finished trace.")

    def _log(self):
        self._log_record[LOG_RECORD_KEY_TIMESTAMP] = time.time()
        self._logger.log(self._log_record)

    def __enter__(self) -> Trace:
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback

        self.end()
