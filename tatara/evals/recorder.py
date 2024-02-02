import atexit
import json
from abc import ABC, abstractmethod

from tatara.evals.eval_types import RecordWithMultipleEvalResults, EvalRun
from tatara.network._tatara_network_client import TataraNetworkClient
from tatara.tatara import _get_network_client

MIN_FLUSH_EVENTS = 100
MIN_FLUSH_SECONDS = 10


class RecorderBase(ABC):
    def __init__(self):
        self.results = []
        # flush_events results on exit
        atexit.register(self.flush_events)


    @abstractmethod
    def flush_events(self):
        pass    
        
    @abstractmethod
    def record_eval_run(self, eval_run):
        pass

class PrintRecorder(RecorderBase):
    def __init__(self):
        super().__init__()

    def record(self, eval_row: RecordWithMultipleEvalResults):
        print(eval_row.to_dict())


class FileRecorder(RecorderBase):
    def __init__(self, event_filepath: str):
        super().__init__()
        self.event_filepath = event_filepath

    def record(self, eval_row: RecordWithMultipleEvalResults):
        self.results.append(eval_row)
        if len(self.results) > MIN_FLUSH_EVENTS:
            self.flush_events()

    def flush_events(self):
        with open(self.event_filepath, "a") as f:
            for eval_row in self.results:
                f.write(json.dumps(eval_row.to_dict()) + "\n")
        self.results = []


class TataraRecorder(RecorderBase):
    def __init__(self):
        super().__init__()
        self.eval_runs = []
        self.total_rows = 0
        self.tatara_network_client: TataraNetworkClient = _get_network_client()


    def flush_events(self):
        for eval_run in self.eval_runs:
            self.tatara_network_client.send_eval_run_post_request(eval_run)
        self.eval_runs = []
    

    def record_eval_run(self, eval_run: EvalRun):
        if self.total_rows > MIN_FLUSH_EVENTS:
            self.flush_events()
        self.eval_runs.append(eval_run)
        self.total_rows += eval_run.num_rows

