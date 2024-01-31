from typing import Any, Dict, Optional
from tatara.tatara_types import (
    ImageFormat,
    DiffusionParams,
    DiffusionPrompt,
    LLMPrompt,
    LLMParams,
    LLMUsageMetrics,
)
from .rating import Rating
from .span import Span


class _EmptySpan(Span):
    @property
    def id(self) -> str:
        return ""

    @property
    def trace_id(self) -> str:
        return ""

    @property
    def event(self) -> str:
        return ""

    def end(self, end_time: Optional[float] = None) -> None:
        pass

    def log_llm_success(
        self,
        prompt: LLMPrompt | str,
        output: str,
        params: Optional[LLMParams] = None,
        usage_metrics: Optional[LLMUsageMetrics] = None,
    ) -> None:
        pass

    def log_diffusion_input(
        self,
        prompt: DiffusionPrompt | str,
        params: Optional[DiffusionParams] = None,
    ) -> None:
        pass

    def log_diffusion_success_with_image_url(
        self,
        image_url: str,
        prompt: DiffusionPrompt | str,
        params: Optional[DiffusionParams] = None,
    ) -> None:
        pass

    def log_diffusion_success_with_image_data(
        self,
        image_data: str,
        image_format: ImageFormat,
        prompt: DiffusionPrompt | str,
        params: Optional[DiffusionParams] = None,
    ) -> None:
        pass

    def log_rating(self, rating: Rating) -> None:
        pass

    def start_span(self, event: str) -> Span:
        return _EmptySpan()

    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        pass

    def __enter__(self) -> Span:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass
