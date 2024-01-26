from network._tatara_network_client import TataraNetworkClient
import os

_tatara_client_state = None


def get_network_client() -> TataraNetworkClient:
    if _tatara_client_state is None:
        raise Exception(
            "Tatara Client State not initialized. Please call init() before using the client."
        )
    return _tatara_client_state.tatara_network_client


def tatara_init():
    global _tatara_client_state
    _tatara_client_state = TataraClientState()


class TataraClientState:
    def __init__(self, is_dev: bool = False):
        if os.environ.get("TATARA_API_KEY") is not None:
            self.api_key = os.environ.get("TATARA_API_KEY")
        else:
            raise ValueError("TATARA_API_KEY environment variable must be set.")
        self.tatara_network_client = TataraNetworkClient(
            api_key=self.api_key, is_dev=is_dev
        )

    # Put logging logic here in a followup PR
    # current_trace: contextvars.ContextVar[Optional[Trace]]
    # current_span: contextvars.ContextVar[Optional[Span]]
    # logger: Optional[_Logger] = field(init=False)
