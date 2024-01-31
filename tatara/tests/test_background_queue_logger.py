from tatara.tatara_logging._background_queue_logger import (
    BackgroundLazyQueueLogger,
)
from unittest.mock import patch
import time
from tatara.network._tatara_network_client import TataraNetworkClient

def _get_logger(max_queue_size=5, flush_interval=60):
    return BackgroundLazyQueueLogger(
        queue_size=max_queue_size, flush_interval=flush_interval,
        tatara_network_client=TataraNetworkClient()
    )


def test_lazy_init():
    logger = _get_logger()
    assert not logger.is_alive


def test_thread_start():
    logger = _get_logger()
    logger._start()
    assert logger.is_alive


def test_log_triggers_start():
    logger = _get_logger()
    logger.log({"message": "test log"})
    assert logger.is_alive


@patch("threading.Thread.start")
def test_log_method(mock_start):
    logger = _get_logger()
    test_log = {"message": "test log"}
    logger.log(test_log)
    assert not logger._log_queue.empty()
    assert logger._log_queue.get() == test_log


@patch(
    "tatara.tatara_logging._background_queue_logger.TataraNetworkClient.send_logs_post_request"
)
def test_sends_logs_when_queue_is_full(mock_send_logs):
    max_queue_size = 5
    logger = _get_logger(max_queue_size=max_queue_size)
    logger._start()

    for i in range(max_queue_size):
        logger.log({"id": f"id_{i}"})

    mock_send_logs.assert_not_called()

    logger.log({"id": "id_last"})

    time.sleep(1)
    mock_send_logs.assert_called()


@patch(
    "tatara.tatara_logging._background_queue_logger.TataraNetworkClient.send_logs_post_request"
)
def test_sends_logs_when_timer_fires(mock_send_logs):
    max_queue_size = 5
    logger = _get_logger(max_queue_size=max_queue_size, flush_interval=1)
    logger._start()

    logger.log({"id": "id_1"})
    mock_send_logs.assert_not_called()

    time.sleep(1.5)
    mock_send_logs.assert_called()


@patch(
    "tatara.tatara_logging._background_queue_logger.TataraNetworkClient.send_logs_post_request"
)
def test_noop_when_timer_fires_on_empty_queue(mock_send_logs):
    logger = _get_logger(flush_interval=1)
    logger._start()
    time.sleep(1.5)

    mock_send_logs.assert_not_called()
