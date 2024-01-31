from .span import Span
from ._background_queue_logger import BackgroundLazyQueueLogger
from tatara.tatara_types import (
    ImageFormat,
    DiffusionParams,
    DiffusionPrompt,
    LLMPrompt,
    LLMParams,
    LLMUsageMetrics,
    LogType,
)
from .rating import Rating
from .utils import _check_event
from typing import Optional, Dict, Any
import uuid
import time
from typing import TYPE_CHECKING

# Imports during type checking only to avoid circular dependencies at runtime.
# _TraceImpl takes a _SpanImpl as an argument, and _SpanImpl takes a _TraceImpl, so
# we need to use TYPE_CHECKING to avoid circular dependencies.
if TYPE_CHECKING:
    from tatara_logging.trace_impl import _TraceImpl

from ._record_keys import (
    LOG_FORMAT_VERSION,
    LOG_RECORD_KEY_EVENT,
    LOG_RECORD_KEY_ID,
    LOG_RECORD_KEY_INTERNAL_ID,
    LOG_RECORD_KEY_PARENT_ID,
    LOG_RECORD_KEY_PARENT_TRACE_ID,
    LOG_RECORD_KEY_PROJECT,
    LOG_RECORD_KEY_PROPERTIES,
    LOG_RECORD_KEY_TIMESTAMP,
    LOG_RECORD_KEY_TYPE,
    LOG_RECORD_KEY_VERSION,
    LOG_RECORD_PROPERTIES_KEY_END_TIME,
    LOG_RECORD_PROPERTIES_KEY_START_TIME,
    LOG_RECORD_PROPERTIES_KEY_LLM_EVENT,
    LOG_RECORD_PROPERTIES_KEY_DIFFUSION_EVENT,
    LOG_RECORD_KEY_METADATA,
)

from tatara.client_state import _get_client_state


class _SpanImpl(Span):
    def __init__(
        self,
        event: str,
        id_: str,
        parent_id: str,
        trace: "_TraceImpl",
        background_queue_logger: BackgroundLazyQueueLogger,
        start_time: Optional[float] = None,
    ):
        self._event = event
        self._internal_id = "s_" + str(uuid.uuid4())
        self._id = id_
        self._parent_id = parent_id
        self._trace = trace

        # self.parent_span_id = parent_span_id
        self._background_queue_logger = background_queue_logger
        now = time.time()
        self._start_time = start_time or now
        self._end_time = None
        self._properties = {
            LOG_RECORD_PROPERTIES_KEY_START_TIME: self._start_time,
            LOG_RECORD_PROPERTIES_KEY_END_TIME: None,
        }
        self._token = _get_client_state().current_span.set(self)

        self._metadata = {}

        self._log_record = {
            LOG_RECORD_KEY_PROJECT: _get_client_state().project,
            LOG_RECORD_KEY_TYPE: LogType.SPAN,
            LOG_RECORD_KEY_TIMESTAMP: now,
            LOG_RECORD_KEY_VERSION: LOG_FORMAT_VERSION,
            LOG_RECORD_KEY_INTERNAL_ID: self._internal_id,
            LOG_RECORD_KEY_ID: self._id,
            LOG_RECORD_KEY_PARENT_TRACE_ID: self._trace.id,
            LOG_RECORD_KEY_PARENT_ID: self._parent_id,
            LOG_RECORD_KEY_EVENT: self._event,
            LOG_RECORD_KEY_PROPERTIES: self._properties,
            LOG_RECORD_KEY_METADATA: self._metadata,
        }

        self._background_queue_logger.log(self._log_record)

    @property
    def id(self) -> str:
        return self._id

    @property
    def trace_id(self):
        return self._trace.id

    @property
    def event(self):
        return self._event

    def _check_finished(self):
        if self._end_time:
            raise RuntimeError("Cannot call method on a finished trace.")

    def _log(self):
        self._log_record[LOG_RECORD_KEY_TIMESTAMP] = time.time()
        self._background_queue_logger.log(self._log_record)

    def log_llm_success(
        self,
        prompt: str | LLMPrompt,
        output: str,
        params: Optional[LLMParams] = None,
        usage_metrics: Optional[LLMUsageMetrics] = None,
    ):
        self._check_finished()

        llm_event = {
            "prompt": prompt,
            "output": output,
            "params": params,
            "usage_metrics": usage_metrics,
        }

        self._properties[LOG_RECORD_PROPERTIES_KEY_LLM_EVENT] = llm_event
        self._log_record[LOG_RECORD_KEY_PROPERTIES] = self._properties
        self._log()

    def log_diffusion_input(
        self,
        prompt: DiffusionPrompt | str,
        params: Optional[DiffusionParams] = None,
    ):
        diffusion_event = {
            "prompt": prompt,
            "params": params,
        }

        self._properties[LOG_RECORD_PROPERTIES_KEY_DIFFUSION_EVENT] = diffusion_event
        self._log_record[LOG_RECORD_KEY_PROPERTIES] = self._properties
        self._log()

    def log_diffusion_success_with_image_url(
        self,
        image_url: str,
        prompt: str | DiffusionPrompt,
        params: Optional[DiffusionParams] = None,
    ):
        self._check_finished()

        diffusion_event = {
            "prompt": prompt,
            "image_url": image_url,
            "params": params,
        }

        self._properties[LOG_RECORD_PROPERTIES_KEY_DIFFUSION_EVENT] = diffusion_event
        self._log_record[LOG_RECORD_KEY_PROPERTIES] = self._properties
        self._log()

    def log_diffusion_success_with_image_data(
        self,
        image_data: str,
        image_format: ImageFormat,
        prompt: str | DiffusionPrompt,
        params: Optional[DiffusionParams] = None,
    ):
        self._check_finished()

        diffusion_event = {
            "image_data": image_data,
            "image_format": image_format,
            "prompt": prompt,
            "params": params,
        }

        self._properties[LOG_RECORD_PROPERTIES_KEY_DIFFUSION_EVENT] = diffusion_event
        self._log_record[LOG_RECORD_KEY_PROPERTIES] = self._properties
        self._log()

    def log_rating(self, rating: Rating) -> None:
        return _get_client_state().log_rating(
            trace_id=self.trace_id, span_event=self.event, rating=rating
        )

    def log_metadata(self, metadata: Dict[str, Any]):
        self._check_finished()

        self._metadata.update(metadata)
        self._log_record[LOG_RECORD_KEY_METADATA] = self._metadata
        self._log()

    def end(self, end_time: Optional[float] = None) -> None:
        self._check_finished()
        _get_client_state().current_span.reset(self._token)

        self._end_time = end_time or time.time()

        self._log_record[LOG_RECORD_KEY_PROPERTIES][
            LOG_RECORD_PROPERTIES_KEY_END_TIME
        ] = self._end_time
        self._log()

    def start_span(self, event: str) -> Span:
        self._check_finished()
        _check_event(event)

        span_id = self._trace._generate_id_for_span_event_and_log(event)

        return _SpanImpl(
            event=event,
            id_=span_id,
            parent_id=self.id,
            trace=self._trace,
            background_queue_logger=self._background_queue_logger,
            start_time=time.time(),
        )

    def __enter__(self) -> Span:
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback

        self.end()
