import atexit
import json
from abc import ABC, abstractmethod

MIN_FLUSH_EVENTS = 100

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

    
    def flush_events(self):
        for eval_run in self.results:
            print(json.dumps(eval_run.to_dict()))
        self.results = []


    def record_eval_run(self, eval_run):
        self.results.append(eval_run)
        self.flush_events()




class FileRecorder(RecorderBase):
    def __init__(self, event_filepath: str):
        super().__init__()
        self.event_filepath = event_filepath

    def record(self, eval_run):
        self.results.append(eval_run)
        if len(self.results) > MIN_FLUSH_EVENTS:
            self.flush_events()

    def flush_events(self):
        with open(self.event_filepath, "a") as f:
            for eval_row in self.results:
                f.write(json.dumps(eval_row.to_dict()) + "\n")
        self.results = []


