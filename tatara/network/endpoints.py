from dataclasses import dataclass

TATARA_SERVER_URL = "https://evals-fastapi.onrender.com"
LOCAL_SERVER_URL = "http://localhost:8080"


@dataclass
class Endpoint:
    server_url: str
    path: str

    def __post_init__(self):
        # Ensure that the path starts with a /
        if not self.path.startswith("/"):
            raise ValueError("Path must start with a /")

    @property
    def url(self) -> str:
        return f"{self.server_url}{self.path}"


class Endpoints:
    def __init__(self, is_dev: bool):
        self.server_url = LOCAL_SERVER_URL if is_dev else TATARA_SERVER_URL

    def _create_endpoint(self, path: str) -> str:
        return Endpoint(
            server_url=self.server_url,
            path=path,
        ).url

    @property
    def log_endpoint(self) -> str:
        return self._create_endpoint("/log")

    @property
    def dataset_endpoint(self) -> str:
        return self._create_endpoint("/dataset")

    @property
    def eval_endpoint(self) -> str:
        return self._create_endpoint("/eval")
