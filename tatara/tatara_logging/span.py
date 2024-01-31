from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from tatara.tatara_types import (
    DiffusionParams,
    DiffusionPrompt,
    LLMPrompt,
    LLMParams,
    LLMUsageMetrics,
    ImageFormat,
)
from .rating import Rating


class Span(ABC):
    @property
    @abstractmethod
    def id(self) -> str:
        """ID of the span. Equivalent to <span.event>_<trace_id>"""

    @property
    @abstractmethod
    def trace_id(self) -> str:
        """ID for the trace that this span belongs to"""

    @property
    @abstractmethod
    def event(self) -> str:
        """Caller provided event name of the span"""

    @abstractmethod
    def end(self, end_time: Optional[float] = None) -> None:
        """Ends the span. No other methods may be called on the span after this."""

    @abstractmethod
    def log_llm_success(
        self,
        prompt: LLMPrompt | str,
        output: str,
        params: Optional[LLMParams] = None,
        usage_metrics: Optional[LLMUsageMetrics] = None,
    ) -> None:
        """Log an LLM event. Only one llm/diffusion call allowed per span."""

    @abstractmethod
    def log_diffusion_input(
        self,
        prompt: DiffusionPrompt | str,
        params: Optional[DiffusionParams] = None,
    ) -> None:
        """Log a diffusion model input event."""

    @abstractmethod
    def log_diffusion_success_with_image_url(
        self,
        image_url: str,
        prompt: DiffusionPrompt | str,
        params: Optional[DiffusionParams] = None,
    ) -> None:
        """Log a diffusion model image creation event. Only one llm/diffusion call allowed per span."""

    @abstractmethod
    def log_diffusion_success_with_image_data(
        self,
        image_data: str,
        image_format: ImageFormat,
        prompt: DiffusionPrompt | str,
        params: Optional[DiffusionParams] = None,
    ) -> None:
        """Log a diffusion model image creation event. `image_data` is expected to be a base64-encoded string. Only one llm/diffusion call allowed per span."""

    @abstractmethod
    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        """Log arbitrary metadata to the span"""

    @abstractmethod
    def log_rating(self, rating: Rating) -> None:
        """Log a rating to the span"""

    @abstractmethod
    def start_span(self, event: str) -> "Span":
        """Start a child span with the current span as the parent and the current trace as the root."""

    @abstractmethod
    def __enter__(self) -> "Span":
        """Enter context manager"""

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit context manager"""
