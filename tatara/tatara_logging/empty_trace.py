from typing import Any, Dict, List, Optional
from tatara_logging.trace import Trace
from tatara_logging.span import Span
from tatara_logging.rating import Rating
from tatara_logging.empty_span import _EmptySpan


class _EmptyTrace(Trace):
    @property
    def id(self) -> str:
        return ""

    @property
    def event(self) -> str:
        return ""

    @property
    def span_ids(self) -> List[str]:
        return []

    def start_span(self, event: str, parent_event: Optional[str]) -> Span:
        return _EmptySpan()

    def end(self) -> None:
        pass

    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        pass

    def log_rating(self, rating: Rating) -> None:
        pass

    def __enter__(self) -> Trace:
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
