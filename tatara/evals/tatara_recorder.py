from .recorder import RecorderBase, MIN_FLUSH_EVENTS
from tatara.network._tatara_network_client import TataraNetworkClient
from tatara.tatara import _get_network_client

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
    

    def record_eval_run(self, eval_run):
        if self.total_rows > MIN_FLUSH_EVENTS:
            self.flush_events()
        self.eval_runs.append(eval_run)
        self.total_rows += eval_run.num_rows

