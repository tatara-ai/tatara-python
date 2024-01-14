from enum import Enum


class ProviderEnum(Enum):
    ANTHROPIC = "anthropic"
    ANYSCALE = "anyscale"
    FIREWORKS = "fireworks"
    HUGGING_FACE = "huggingface"
    OPENAI = "openai"
    STABILITYAI = "stabilityai"
    TOGETHERAI = "togetherai"
    OTHER = "other"
