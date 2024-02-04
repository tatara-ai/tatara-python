import atexit
import json
from abc import ABC, abstractmethod

from tatara.evals.eval_types import RecordWithEvalResults

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

    def record(self, eval_row: RecordWithEvalResults):
        print(eval_row.to_dict())


class FileRecorder(RecorderBase):
    def __init__(self, event_filepath: str):
        super().__init__()
        self.event_filepath = event_filepath

    def record(self, eval_row: RecordWithEvalResults):
        self.results.append(eval_row)
        if len(self.results) > MIN_FLUSH_EVENTS:
            self.flush_events()

    def flush_events(self):
        with open(self.event_filepath, "a") as f:
            for eval_row in self.results:
                f.write(json.dumps(eval_row.to_dict()) + "\n")
        self.results = []


