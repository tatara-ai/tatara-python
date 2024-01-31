from typing import Any, Dict, List, Optional
from .trace import Trace
from .span import Span
from .rating import Rating
from .empty_span import _EmptySpan


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
