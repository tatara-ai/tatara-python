from abc import ABC, abstractmethod
from .rating import Rating
from typing import Any, Dict, List, Optional
from .span import Span


class Trace(ABC):
    @property
    @abstractmethod
    def id(self) -> str:
        """ID of the trace. If the user does not provide an ID, equivalent to trace_id"""

    @property
    @abstractmethod
    def event(self) -> str:
        """Caller provided event name of the trace"""

    @property
    @abstractmethod
    def span_ids(self) -> List[str]:
        """List of spans associated with the trace"""

    @abstractmethod
    def start_span(self, event: str, parent_event: Optional[str]) -> Span:
        """Start a span. Optionally provide a parent span event. If none is provided, the current trace is used as the parent."""

    @abstractmethod
    def end(self) -> None:
        """End the trace"""

    @abstractmethod
    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        """Log arbitrary metadata to the trace"""

    @abstractmethod
    def log_rating(self, rating: Rating) -> None:
        """Log a rating to the trace"""

    @abstractmethod
    def __enter__(self) -> "Trace":
        """Enter context manager"""

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit context manager"""
