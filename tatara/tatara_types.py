from dataclasses import dataclass
from typing import Optional, Dict
from .provider_enum import ProviderEnum
from enum import Enum


@dataclass
class LLMUsageMetrics:
    prompt_tokens: int
    completion_tokens: int

    @classmethod
    def from_oai_completion_usage(cls, completion_usage): # openai.types.CompletionUsage
        return cls(
            prompt_tokens=completion_usage.prompt_tokens,
            completion_tokens=completion_usage.completion_tokens,
        )


@dataclass
class LLMPrompt:
    system_prompt: Optional[str] = None
    prompt_template: Optional[str] = None
    input_variables: Optional[Dict[str, str]] = None


@dataclass
class LLMParams:
    frequency_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    model: Optional[str] = None
    provider: Optional[str | ProviderEnum] = None


@dataclass
class DiffusionPrompt:
    prompt_template: Optional[str] = None
    input_variables: Optional[Dict[str, str]] = None
    negative_prompt: Optional[str] = None


@dataclass
class DiffusionParams:
    steps: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    model: Optional[str] = None
    provider: Optional[str | ProviderEnum] = None


class LogType(Enum):
    TRACE = 0
    SPAN = 1
    RATING = 2
    EVAL = 3


class ImageFormat(Enum):
    PNG = "png"
    JPG = "jpg"
