import atexit
import json
from abc import ABC, abstractmethod

from evals.eval_types import EvalRun
from network._tatara_network_client import TataraNetworkClient

MIN_FLUSH_EVENTS = 100
MIN_FLUSH_SECONDS = 10


class RecorderBase(ABC):
    def __init__(self):
        self.results = []
        # flush_events results on exit
        atexit.register(self.flush_events)

    @abstractmethod
    def _flush_internal(self, results):
        pass

    @abstractmethod
    def flush_events(self):
        pass

    @abstractmethod
    def record_eval_row(self, eval_row):
        pass

    @abstractmethod
    def record_eval_run(self, eval_run):
        pass


class FileRecorder(RecorderBase):
    def __init__(self, event_filepath: str):
        super().__init__()
        self.event_filepath = event_filepath

    def record(self, eval_row: EvalRun):
        self.results.append(eval_row)
        if len(self.results) > MIN_FLUSH_EVENTS:
            self.flush_events()

    def flush_events(self):
        with open(self.event_filepath, "a") as f:
            for eval_row in self.results:
                f.write(json.dumps(eval_row.to_dict()) + "\n")
        self.results = []


class TataraRecorder(RecorderBase):
    def __init__(self, tatara_network_client: TataraNetworkClient):
        super().__init__()
        self.eval_rows = []
        self.tatara_network_client = tatara_network_client

    def record_eval_run(self, eval_run: EvalRun):
        self.tatara_network_client.send_eval_run_post_request(eval_run)
