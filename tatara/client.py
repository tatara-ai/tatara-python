import time
from inspect import iscoroutinefunction
from typing import Optional
import wrapt
import logging
from tatara_logging.trace import Trace
from tatara_logging.span import Span
from tatara_logging.empty_span import _EmptySpan
from tatara_logging.empty_trace import _EmptyTrace
from tatara_logging.trace_impl import _TraceImpl
from tatara_logging.rating import Rating
from tatara.types import DiffusionPrompt, DiffusionParams, LogType, ImageFormat
from tatara_logging.utils import _gen_id_from_trace_and_event
from tatara_logging._record_keys import (
    LOG_RECORD_KEY_ID,
    LOG_RECORD_KEY_PROJECT,
    LOG_RECORD_KEY_PROPERTIES,
    LOG_RECORD_KEY_TIMESTAMP,
    LOG_RECORD_KEY_TYPE,
    LOG_RECORD_PROPERTIES_KEY_DIFFUSION_EVENT,
)
import os

from client_state import TataraClientState, _get_client_state


def init(project: str, api_key: Optional[str] = None):
    if api_key is None:
        if os.environ.get("TATARA_API_KEY") is not None:
            api_key = os.environ.get("TATARA_API_KEY")
        else:
            raise ValueError("TATARA_API_KEY environment variable must be set.")

    global _tatara_client_state
    _tatara_client_state = TataraClientState(
        project,
        api_key,
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
