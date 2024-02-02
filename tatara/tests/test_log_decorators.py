from unittest.mock import patch

import pytest

from tatara.tatara import log_span, log_trace


@log_trace(event="sync_event")
def sync_trace_function(x, y):
    return x + y


@log_trace(event="async_event")
async def async_trace_function(x, y):
    return x * y


@patch("tatara.tatara.start_trace")
def test_log_trace_sync(mock_start_trace):
    sync_trace_function(2, 3)
    mock_start_trace.assert_called_once_with(event="sync_event", id_=None, user_id=None)


@patch("tatara.tatara.start_trace")
def test_log_trace_sync_with_id(mock_start_trace):
    sync_trace_function(2, 3, trace_id="test_id")  # type: ignore
    mock_start_trace.assert_called_once_with(
        event="sync_event", id_="test_id", user_id=None
    )


@patch("tatara.tatara.start_trace")
def test_log_trace_sync_with_user_id(mock_start_trace):
    sync_trace_function(2, 3, tatara_user_id="test_user_id")  # type: ignore
    mock_start_trace.assert_called_once_with(
        event="sync_event", id_=None, user_id="test_user_id"
    )


@patch("tatara.tatara.start_trace")
def test_log_trace_sync_with_id_and_user_id(mock_start_trace):
    sync_trace_function(2, 3, trace_id="test_id", tatara_user_id="test_user_id")  # type: ignore
    mock_start_trace.assert_called_once_with(
        event="sync_event", id_="test_id", user_id="test_user_id"
    )


def test_log_trace_fails_without_event():
    with pytest.raises(TypeError):

        @log_trace  # type: ignore
        def f(x, y):
            return x + y

        f(2, 3)


@patch("tatara.tatara.start_trace")
@pytest.mark.asyncio
async def test_log_trace_async(mock_start_trace):
    await async_trace_function(2, 3)
    mock_start_trace.assert_called_once_with(
        event="async_event", id_=None, user_id=None
    )


@log_span(event="sync_event")
def sync_span_function(x, y):
    return x + y


@log_span(event="async_event")
async def async_span_function(x, y):
    return x * y


@patch("tatara.tatara.start_span")
def test_log_span_sync(mock_start_span):
    sync_span_function(2, 3)
    mock_start_span.assert_called_once_with("sync_event", None)


@patch("tatara.tatara.start_span")
def test_log_span_sync_with_parent_event(mock_start_span):
    @log_span(event="sync_event", parent_event="parent_event")
    def f(x, y):
        return x + y

    f(2, 3)
    mock_start_span.assert_called_once_with("sync_event", "parent_event")


def test_log_span_fails_without_event():
    with pytest.raises(TypeError):

        @log_span  # type: ignore
        def f(x, y):
            return x + y

        f(2, 3)


@patch("tatara.tatara.start_span")
def test_log_span_with_caller_event(mock_start_span):
    @log_span  # type: ignore
    def f(x, y):
        return x + y

    f(2, 3, tatara_event="caller_event")  # type: ignore
    mock_start_span.assert_called_once_with("caller_event", None)


@patch("tatara.tatara.start_span")
def test_log_span_with_caller_event_and_parent(mock_start_span):
    @log_span  # type: ignore
    def f(x, y):
        return x + y

    f(2, 3, tatara_event="caller_event", tatara_parent_event="caller_parent_event")  # type: ignore
    mock_start_span.assert_called_once_with("caller_event", "caller_parent_event")


@patch("tatara.tatara.start_span")
@pytest.mark.asyncio
async def test_async(mock_start_span):
    await async_span_function(2, 3)
    mock_start_span.assert_called_once_with("async_event", None)
