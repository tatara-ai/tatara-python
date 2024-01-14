import contextvars
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from inspect import iscoroutinefunction
from typing import Any, Dict, List, Optional
import wrapt
import logging
from openai.types import CompletionUsage
from src.tatara._background_queue_logger import BackgroundLazyQueueLogger
from src.tatara.provider_enum import ProviderEnum


class ImageFormat(Enum):
    PNG = "png"
    JPG = "jpg"


@dataclass
class LLMUsageMetrics:
    prompt_tokens: int
    completion_tokens: int

    @classmethod
    def from_oai_completion_usage(cls, completion_usage: CompletionUsage):
        return cls(
            prompt_tokens=completion_usage.prompt_tokens,
            completion_tokens=completion_usage.completion_tokens,
        )


@dataclass
class LLMPrompt:
    system_prompt: Optional[str] = None
    prompt_template: Optional[str] = None
    input_variables: Optional[Dict[str, str]] = None


@dataclass
class LLMParams:
    frequency_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    model: Optional[str] = None
    provider: Optional[str | ProviderEnum] = None


@dataclass
class DiffusionPrompt:
    prompt_template: Optional[str] = None
    input_variables: Optional[Dict[str, str]] = None
    negative_prompt: Optional[str] = None


@dataclass
class DiffusionParams:
    steps: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    model: Optional[str] = None
    provider: Optional[str | ProviderEnum] = None


class BinaryRating(Enum):
    UPVOTE = 1
    NOVOTE = 0
    DOWNVOTE = -1


@dataclass
class Rating:
    rating: BinaryRating  # TODO: support other rating types (i.e. star scale, slider)
    feedback: Optional[str] = None


class Span(ABC):
    @property
    @abstractmethod
    def id(self) -> str:
        """ID of the span. Equivalent to <span.event>_<trace_id>"""

    @property
    @abstractmethod
    def trace_id(self) -> str:
        """ID for the trace that this span belongs to"""

    @property
    @abstractmethod
    def event(self) -> str:
        """Caller provided event name of the span"""

    @abstractmethod
    def end(self, end_time: Optional[float] = None) -> None:
        """Ends the span. No other methods may be called on the span after this."""

    @abstractmethod
    def log_llm_success(
        self,
        prompt: LLMPrompt | str,
        output: str,
        params: Optional[LLMParams] = None,
        usage_metrics: Optional[LLMUsageMetrics] = None,
    ) -> None:
        """Log an LLM event. Only one llm/diffusion call allowed per span."""

    @abstractmethod
    def log_diffusion_input(
        self,
        prompt: DiffusionPrompt | str,
        params: Optional[DiffusionParams] = None,
    ) -> None:
        """Log a diffusion model input event."""

    @abstractmethod
    def log_diffusion_success_with_image_url(
        self,
        image_url: str,
        prompt: DiffusionPrompt | str,
        params: Optional[DiffusionParams] = None,
    ) -> None:
        """Log a diffusion model image creation event. Only one llm/diffusion call allowed per span."""

    @abstractmethod
    def log_diffusion_success_with_image_data(
        self,
        image_data: str,
        image_format: ImageFormat,
        prompt: DiffusionPrompt | str,
        params: Optional[DiffusionParams] = None,
    ) -> None:
        """Log a diffusion model image creation event. `image_data` is expected to be a base64-encoded string. Only one llm/diffusion call allowed per span."""

    @abstractmethod
    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        """Log arbitrary metadata to the span"""

    @abstractmethod
    def log_rating(self, rating: Rating) -> None:
        """Log a rating to the span"""

    @abstractmethod
    def start_span(self, event: str) -> "Span":
        """Start a child span with the current span as the parent and the current trace as the root."""

    @abstractmethod
    def __enter__(self) -> "Span":
        """Enter context manager"""

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit context manager"""


class Trace(ABC):
    @property
    @abstractmethod
    def id(self) -> str:
        """ID of the trace. If the user does not provide an ID, equivalent to trace_id"""

    @property
    @abstractmethod
    def event(self) -> str:
        """Caller provided event name of the trace"""

    @property
    @abstractmethod
    def span_ids(self) -> List[str]:
        """List of spans associated with the trace"""

    @abstractmethod
    def start_span(self, event: str, parent_event: Optional[str]) -> Span:
        """Start a span. Optionally provide a parent span event. If none is provided, the current trace is used as the parent."""

    @abstractmethod
    def end(self) -> None:
        """End the trace"""

    @abstractmethod
    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        """Log arbitrary metadata to the trace"""

    @abstractmethod
    def log_rating(self, rating: Rating) -> None:
        """Log a rating to the trace"""

    @abstractmethod
    def __enter__(self) -> "Trace":
        """Enter context manager"""

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit context manager"""


DEFAULT_QUEUE_SIZE = 1000
DEFAULT_FLUSH_INTERVAL = 60.0


def init(
    project: str,
    queue_size: int = DEFAULT_QUEUE_SIZE,
    flush_interval: float = DEFAULT_FLUSH_INTERVAL,
    api_key: Optional[str] = None,
):
    """
    Initialize the global tracer with the provided API key and project event.
    """
    global _logger
    _logger = _Logger(
        api_key=api_key,
        project=project,
        queue_size=queue_size,
        flush_interval=flush_interval,
    )


def current_trace() -> Trace:
    """
    Get the current trace. If there is no current trace, returns an empty trace.
    """
    current_trace = _logger.current_trace.get()

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
    current_span = _logger.current_span.get()

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
    return _logger.create_trace(event, id_, user_id)


def start_span(event: str, parent_event: Optional[str] = None) -> Span:
    """
    Create a new span. Automatically finds the current trace and returns an empty span an active trace is not found.
    """
    current_trace = _logger.current_trace.get()

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
        LOG_RECORD_KEY_PROJECT: _logger.project,
        LOG_RECORD_KEY_TYPE: LogType.SPAN,
        LOG_RECORD_KEY_TIMESTAMP: time.time(),
        LOG_RECORD_KEY_PROPERTIES: {
            LOG_RECORD_PROPERTIES_KEY_DIFFUSION_EVENT: {
                "prompt": prompt,
                "params": params,
            }
        },
    }
    return _logger.bglq_logger.log(log)


def log_diffusion_output_with_image_data(
    trace_id: str, event: str, image_data: str, image_format: ImageFormat
):
    pass


def log_diffusion_output_with_image_url(image_url: str, trace_id: str, event: str):
    log = {
        LOG_RECORD_KEY_ID: _gen_id_from_trace_and_event(trace_id, event),
        LOG_RECORD_KEY_PROJECT: _logger.project,
        LOG_RECORD_KEY_TYPE: LogType.SPAN,
        LOG_RECORD_KEY_TIMESTAMP: time.time(),
        LOG_RECORD_KEY_PROPERTIES: {
            LOG_RECORD_PROPERTIES_KEY_DIFFUSION_EVENT: {
                "image_url": image_url,
            }
        },
    }
    return _logger.bglq_logger.log(log)


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
    _logger.log_rating(rating=rating, trace_id=trace_id, span_event=span_event)


def log_rating_for_trace(id: str, rating: Rating):
    """
    Add a rating for a trace. ID can be either the user-provided id or the trace_id.
    """
    _logger.log_rating(rating=rating, trace_id=id, span_event=None)


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


####################################################################
## INTERNALS BELOW                                                ##
####################################################################

from ._record_keys import (  # noqa: E402
    LOG_FORMAT_VERSION,
    LOG_RECORD_KEY_EVENT,
    LOG_RECORD_KEY_HAS_RATING,
    LOG_RECORD_KEY_ID,
    LOG_RECORD_KEY_INTERNAL_ID,
    LOG_RECORD_KEY_METADATA,
    LOG_RECORD_KEY_PARENT_ID,
    LOG_RECORD_KEY_PARENT_TRACE_ID,
    LOG_RECORD_KEY_PROJECT,
    LOG_RECORD_KEY_PROPERTIES,
    LOG_RECORD_KEY_RATING,
    LOG_RECORD_KEY_SPANS,
    LOG_RECORD_KEY_TIMESTAMP,
    LOG_RECORD_KEY_TYPE,
    LOG_RECORD_KEY_USER_ID,
    LOG_RECORD_KEY_VERSION,
    LOG_RECORD_PROPERTIES_KEY_DIFFUSION_EVENT,
    LOG_RECORD_PROPERTIES_KEY_END_TIME,
    LOG_RECORD_PROPERTIES_KEY_LLM_EVENT,
    LOG_RECORD_PROPERTIES_KEY_START_TIME,
)


def _gen_id_from_trace_and_event(trace_id: str, event: Optional[str]) -> str:
    if event:
        return event + "_" + trace_id
    return trace_id


class LogType(Enum):
    TRACE = 0
    SPAN = 1
    RATING = 2


class _SpanImpl(Span):
    def __init__(
        self,
        event: str,
        id_: str,
        parent_id: str,
        trace: "_TraceImpl",
        logger: BackgroundLazyQueueLogger,
        start_time: Optional[float] = None,
    ):
        self._event = event
        self._internal_id = "s_" + str(uuid.uuid4())
        self._id = id_
        self._parent_id = parent_id
        self._trace = trace

        # self.parent_span_id = parent_span_id
        self._logger = logger
        now = time.time()
        self._start_time = start_time or now
        self._end_time = None
        self._properties = {
            LOG_RECORD_PROPERTIES_KEY_START_TIME: self._start_time,
            LOG_RECORD_PROPERTIES_KEY_END_TIME: None,
        }
        self._token = _logger.current_span.set(self)

        self._metadata = {}

        self._log_record = {
            LOG_RECORD_KEY_PROJECT: _logger.project,
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

        self._logger.log(self._log_record)

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
        self._logger.log(self._log_record)

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
        return _logger.log_rating(
            trace_id=self.trace_id, span_event=self.event, rating=rating
        )

    def log_metadata(self, metadata: Dict[str, Any]):
        self._check_finished()

        self._metadata.update(metadata)
        self._log_record[LOG_RECORD_KEY_METADATA] = self._metadata
        self._log()

    def end(self, end_time: Optional[float] = None) -> None:
        self._check_finished()
        _logger.current_span.reset(self._token)

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
            logger=self._logger,
            start_time=time.time(),
        )

    def __enter__(self) -> Span:
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback

        self.end()


class _EmptySpan(Span):
    @property
    def id(self) -> str:
        return ""

    @property
    def trace_id(self) -> str:
        return ""

    @property
    def event(self) -> str:
        return ""

    def end(self, end_time: Optional[float] = None) -> None:
        pass

    def log_llm_success(
        self,
        prompt: LLMPrompt | str,
        output: str,
        params: Optional[LLMParams] = None,
        usage_metrics: Optional[LLMUsageMetrics] = None,
    ) -> None:
        pass

    def log_diffusion_input(
        self,
        prompt: DiffusionPrompt | str,
        params: Optional[DiffusionParams] = None,
    ) -> None:
        pass

    def log_diffusion_success_with_image_url(
        self,
        image_url: str,
        prompt: DiffusionPrompt | str,
        params: Optional[DiffusionParams] = None,
    ) -> None:
        pass

    def log_diffusion_success_with_image_data(
        self,
        image_data: str,
        image_format: ImageFormat,
        prompt: DiffusionPrompt | str,
        params: Optional[DiffusionParams] = None,
    ) -> None:
        pass

    def log_rating(self, rating: Rating) -> None:
        pass

    def start_span(self, event: str) -> Span:
        return _EmptySpan()

    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        pass

    def __enter__(self) -> Span:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass


class _EmptyTrace(Trace):
    @property
    def id(self) -> str:
        return ""

    @property
    def event(self) -> str:
        return ""

    @property
    def span_ids(self) -> List[str]:
        return []

    def start_span(self, event: str, parent_event: Optional[str]) -> Span:
        return _EmptySpan()

    def end(self) -> None:
        pass

    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        pass

    def log_rating(self, rating: Rating) -> None:
        pass

    def __enter__(self) -> Trace:
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def _check_event(event: str) -> None:
    if len(event) > 64:
        raise ValueError(
            f"'{event}' is too long of a event. Must be under 64 characters."
        )
    if not re.match(r"^\w+$", event):
        raise ValueError(
            f"'{event}' is an invalid event. Must consist of only alphanumeric characters and underscores."
        )


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
        self._token = _logger.current_trace.set(self)

        self._properties = {
            LOG_RECORD_PROPERTIES_KEY_START_TIME: self._start_time,
            LOG_RECORD_PROPERTIES_KEY_END_TIME: None,
        }

        self._log_record = {
            LOG_RECORD_KEY_PROJECT: _logger.project,
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
            logger=self._logger,
            start_time=time.time(),
        )

    def end(self, end_time: Optional[float] = None):
        self._check_finished()

        self._end_time = end_time or time.time()
        _logger.current_trace.reset(self._token)

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
        return log_rating_for_trace(self.id, rating)

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


class _Logger:
    current_trace: contextvars.ContextVar[Optional[Trace]]
    current_span: contextvars.ContextVar[Optional[Span]]

    def __init__(
        self,
        project: str,
        queue_size: int,
        flush_interval: float,
        api_key: Optional[str] = None,
    ):
        self._api_key = api_key
        self.project = project
        self.current_trace = contextvars.ContextVar("current_trace", default=None)
        self.current_span = contextvars.ContextVar("current_span", default=None)
        self.bglq_logger = BackgroundLazyQueueLogger(
            queue_size, flush_interval=flush_interval, api_key=api_key
        )

    def create_trace(
        self, event: str, id_: Optional[str] = None, user_id: Optional[str] = None
    ) -> Trace:
        return _TraceImpl(event, logger=self.bglq_logger, id_=id_, user_id=user_id)

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
            self.bglq_logger.log(
                {
                    LOG_RECORD_KEY_PROJECT: self.project,
                    LOG_RECORD_KEY_TYPE: LogType.TRACE,
                    LOG_RECORD_KEY_ID: id,
                    LOG_RECORD_KEY_TIMESTAMP: now,
                    LOG_RECORD_KEY_VERSION: LOG_FORMAT_VERSION,
                    LOG_RECORD_KEY_HAS_RATING: 1,
                }
            )
