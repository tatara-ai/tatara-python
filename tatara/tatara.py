from inspect import iscoroutinefunction
from typing import Optional, List, Dict, Any
import wrapt
import logging
from .tatara_logging.trace import Trace
from .tatara_logging.span import Span
from .tatara_logging.empty_span import _EmptySpan
from .tatara_logging.empty_trace import _EmptyTrace
from .tatara_logging.rating import Rating
from .tatara_types import DiffusionPrompt, DiffusionParams, LogType, ImageFormat, LLMPrompt, LLMParams, LLMUsageMetrics
from .tatara_logging.utils import _gen_id_from_trace_and_event
from .tatara_logging._record_keys import (
    LOG_RECORD_KEY_ID,
    LOG_RECORD_KEY_PROJECT,
    LOG_RECORD_KEY_EVENT,
    LOG_RECORD_KEY_INTERNAL_ID,
    LOG_RECORD_KEY_SPANS,
    LOG_RECORD_KEY_USER_ID,
    LOG_FORMAT_VERSION,
    LOG_RECORD_KEY_PARENT_ID,
    LOG_RECORD_KEY_PARENT_TRACE_ID,
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

import time
from .tatara_logging._background_queue_logger import BackgroundLazyQueueLogger
from .tatara_logging.utils import _check_event
import uuid
import os

from .client_state import TataraClientState
from .network._tatara_network_client import TataraNetworkClient


DEFAULT_QUEUE_SIZE = 1000
DEFAULT_FLUSH_INTERVAL = 60.0

_tatara_client_state: Optional[TataraClientState] = None

def init(project: str, queue_size: int = DEFAULT_QUEUE_SIZE, flush_interval: float = DEFAULT_FLUSH_INTERVAL, api_key: Optional[str] = None, is_dev: Optional[bool] = False):
    if api_key is None:
        if os.environ.get("TATARA_API_KEY") is not None:
            api_key = os.environ.get("TATARA_API_KEY")
        else:
            raise ValueError("TATARA_API_KEY environment variable must be set.")

    global _tatara_client_state
    _tatara_client_state = TataraClientState(
        project,
        queue_size = queue_size,
        flush_interval = flush_interval,
        api_key=api_key,
        is_dev=is_dev if is_dev is not None else False,
    )


def current_trace() -> Trace:
    """
    Get the current trace. If there is no current trace, returns an empty trace.
    """
    current_trace = _get_client_state().current_trace.get()

    if current_trace is None:
        logging.log(
            logging.WARN, "No current trace found. Did you forget to start the trace?"
        )
        return _EmptyTrace()

    return current_trace


def current_span() -> Span:
    """
    Get the current trace. If there is no current trace, returns an empty trace.
    """
    current_span = _get_client_state().current_span.get()

    if current_span is None:
        logging.log(
            logging.WARN, "No current span found. Did you forget to start the span?"
        )
        return _EmptySpan()

    return current_span


def rate_trace(trace_id: str, rating: Rating):
    """
    Rate a trace.
    """
    # TODO: trace rating
    pass


def start_trace(
    event: str, id_: Optional[str] = None, user_id: Optional[str] = None
) -> Trace:
    """
    Create a new trace.
    """
    return _TraceImpl(
        event, logger=_get_client_state().bglq_logger, id_=id_, user_id=user_id
    )


def start_span(event: str, parent_event: Optional[str] = None) -> Span:
    """
    Create a new span. Automatically finds the current trace and returns an empty span an active trace is not found.
    """
    current_trace = _get_client_state().current_trace.get()

    if current_trace is None:
        return _EmptySpan()

    return current_trace.start_span(event, parent_event)


def log_diffusion_input(
    trace_id: str,
    event: str,
    prompt: DiffusionPrompt | str,
    params: Optional[DiffusionParams] = None,
):
    log = {
        LOG_RECORD_KEY_ID: _gen_id_from_trace_and_event(trace_id, event),
        LOG_RECORD_KEY_PROJECT: _get_client_state().project,
        LOG_RECORD_KEY_TYPE: LogType.SPAN,
        LOG_RECORD_KEY_TIMESTAMP: time.time(),
        LOG_RECORD_KEY_PROPERTIES: {
            LOG_RECORD_PROPERTIES_KEY_DIFFUSION_EVENT: {
                "prompt": prompt,
                "params": params,
            }
        },
    }
    return _get_client_state().bglq_logger.log(log)


def log_diffusion_output_with_image_data(
    trace_id: str, event: str, image_data: str, image_format: ImageFormat
):
    pass


def log_diffusion_output_with_image_url(image_url: str, trace_id: str, event: str):
    log = {
        LOG_RECORD_KEY_ID: _gen_id_from_trace_and_event(trace_id, event),
        LOG_RECORD_KEY_PROJECT: _get_client_state().project,
        LOG_RECORD_KEY_TYPE: LogType.SPAN,
        LOG_RECORD_KEY_TIMESTAMP: time.time(),
        LOG_RECORD_KEY_PROPERTIES: {
            LOG_RECORD_PROPERTIES_KEY_DIFFUSION_EVENT: {
                "image_url": image_url,
            }
        },
    }
    return _get_client_state().bglq_logger.log(log)


def log_diffusion_call_with_image_data(
    image_data: str,
    image_format: ImageFormat,
    trace_id: str,
    event: str,
    prompt: DiffusionPrompt | str,
    params: Optional[DiffusionParams] = None,
):
    pass


def log_diffusion_call_with_image_url(
    image_url: str,
    trace_id: str,
    event: str,
    prompt: DiffusionPrompt | str,
    params: Optional[DiffusionParams] = None,
):
    pass


def log_rating_for_span(span_event: str, trace_id: str, rating: Rating):
    """
    Add a rating for a span.
    """
    _get_client_state().log_rating(
        rating=rating, trace_id=trace_id, span_event=span_event
    )


def log_rating_for_trace(id: str, rating: Rating):
    """
    Add a rating for a trace. ID can be either the user-provided id or the trace_id.
    """
    _get_client_state().log_rating(rating=rating, trace_id=id, span_event=None)


def log_trace(event: str):
    if callable(event):
        raise TypeError(
            "The 'log_trace' decorator expects an 'event' argument. Usage: @log_trace(event='your_event')"
        )

    def wrapper(wrapped):
        @wrapt.decorator
        async def _async_trace_wrapper(wrapped, instance, args, kwargs):
            trace_id = kwargs.pop("trace_id", None)
            user_id = kwargs.pop("tatara_user_id", None)

            with start_trace(event=event, id_=trace_id, user_id=user_id):
                return await wrapped(*args, **kwargs)

        @wrapt.decorator
        def _trace_wrapper(wrapped, instance, args, kwargs):
            trace_id = kwargs.pop("trace_id", None)
            user_id = kwargs.pop("tatara_user_id", None)

            with start_trace(event=event, id_=trace_id, user_id=user_id):
                return wrapped(*args, **kwargs)

        if iscoroutinefunction(wrapped):
            return _async_trace_wrapper(wrapped)  # type: ignore
        else:
            return _trace_wrapper(wrapped)  # type: ignore

    return wrapper


def log_span(event: Optional[str] = None, parent_event: Optional[str] = None):
    if callable(event):
        # Decorator was used without arguments, and 'event' is the function to be decorated.
        func = event
        return _create_span_decorator(event=None, parent_event=None)(func)
    else:
        # Decorator was used with string arguments.
        def decorator(func):
            return _create_span_decorator(event=event, parent_event=parent_event)(func)

        return decorator


def _create_span_decorator(event: Optional[str], parent_event: Optional[str] = None):
    def _check_event(event: Optional[str]):
        if event is None:
            raise TypeError(
                """Missing required 'event' argument. The 'log_span' decorator expects either:

1) An 'event' argument directly passed in to the decorator: 

    @log_span(event='your_event')
    def decorated_function():
        # do stuff

2) A 'tatara_event' argument passed from the caller of the decorated function:

    @log_span
    def decorated_function():
        # do stuff
    
    decorated_function(tatara_event='your_event') 
"""
            )

    def wrapper(wrapped):
        @wrapt.decorator
        async def _async_span_wrapper(wrapped, instance, args, kwargs):
            _event = kwargs.pop("tatara_event", event)
            _check_event(_event)
            _parent_event = kwargs.pop("tatara_parent_event", parent_event)
            with start_span(_event, _parent_event) as span:  # noqa: F841
                return await wrapped(*args, **kwargs)

        @wrapt.decorator
        def _span_wrapper(wrapped, instance, args, kwargs):
            _event = kwargs.pop("tatara_event", event)
            _check_event(_event)
            _parent_event = kwargs.pop("tatara_parent_event", parent_event)
            with start_span(_event, _parent_event) as span:  # noqa: F841
                return wrapped(*args, **kwargs)

        if iscoroutinefunction(wrapped):
            return _async_span_wrapper(wrapped)  # type: ignore
        else:
            return _span_wrapper(wrapped)  # type: ignore

    return wrapper

def _get_client_state() -> TataraClientState:
    if _tatara_client_state is None:
        raise Exception(
            "Tatara Client State not initialized. Please call init() before using the client."
        )
    return _tatara_client_state


def _get_network_client() -> TataraNetworkClient:
    if _tatara_client_state is None:
        raise Exception(
            "Tatara Client State not initialized. Please call init() before using the client."
        )
    return _tatara_client_state.tatara_network_client

### IMPLEMENTATION

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

