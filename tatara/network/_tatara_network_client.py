import requests
from requests.adapters import HTTPAdapter
from typing import Optional, Dict, List, Any
import json
from .endpoints import Endpoints
from requests.models import Response


class TataraNetworkClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        is_dev: bool = False,
    ):
        self._headers = {"Authorization": api_key, "Content-Type": "application/json"}
        self.endpoints = Endpoints(is_dev=is_dev)

        adapter = HTTPAdapter(max_retries=0)
        self._http = requests.Session()
        self._http.mount("https://", adapter)
        self._http.mount("http://", adapter)

    ### LOGS
    def send_logs_post_request(self, logs: str) -> Optional[Response]:
        return self.send_post_request(f"{self.endpoints.log_endpoint}/write", logs)

    ### DATASETS
    def send_create_dataset_post_request(self, dataset_name: str) -> Optional[Response]:
        return self.send_post_request(
            self.endpoints.dataset_endpoint, data=json.dumps(dataset_name)
        )

    def send_dataset_get_request(self, dataset_name: str) -> Response:
        return self.send_get_request(
            f"{self.endpoints.dataset_endpoint}/{dataset_name}"
        )

    def send_insert_records_post_request(
        self, dataset_name: str, records: List[Dict[str, Any]]
    ) -> Optional[Response]:
        return self.send_post_request(
            f"{self.endpoints.dataset_endpoint}/{dataset_name}/insert_records",
            data=json.dumps(records),
        )

    def send_attach_records_post_request(
        self, dataset_name: str, record_ids: List[str]
    ) -> Optional[Response]:
        return self.send_post_request(
            f"{self.endpoints.dataset_endpoint}/{dataset_name}/attach_records",
            data=json.dumps(record_ids),
        )

    ### EVALS
    def send_eval_run_post_request(self, eval_run) -> Optional[Response]:
        return self.send_post_request(
            f"{self.endpoints.eval_endpoint}/write", data=json.dumps(eval_run.to_dict())
        )

    ### GENERIC REQUESTS
    def send_post_request(self, endpoint: str, data: str) -> Optional[Response]:
        response = self._http.request(
            "POST",
            endpoint,
            data=data,
            headers=self._headers,
        )
        response.raise_for_status()
        return response

    def send_get_request(
        self, endpoint: str, params: Optional[dict] = None
    ) -> Response:
        response = self._http.get(
            endpoint,
            headers=self._headers,
            params=params,
        )
        response.raise_for_status()
        return response
