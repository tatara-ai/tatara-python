import urllib3
from urllib3.util.retry import Retry
from typing import Optional
import os

TATARA_API_ENDPOINT = "https://evals-fastapi.onrender.com/log/write"


class NetworkLogger:
    def __init__(
        self, api_key: Optional[str] = None, endpoint_url: str = TATARA_API_ENDPOINT
    ):
        if api_key is None:
            api_key = os.getenv("TATARA_API_KEY")

        if api_key is None:
            raise Exception("TATARA_API_KEY not set")

        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST"],
        )
        self._http = urllib3.PoolManager(
            headers={"Authorization": api_key, "Content-Type": "application/json"},
            retries=retry_strategy,
        )
        self._endpoint_url = endpoint_url

    # TODO: cleanup and make this agnostic of the endpoint. i.e. send arbitrary POST
    def send_logs_post_request(self, logs: str):
        try:
            response = self._http.request(
                "POST",
                self._endpoint_url,
                body=logs,
            )
            return response
        except urllib3.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            return None
