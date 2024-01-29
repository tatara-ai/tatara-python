from dataclasses import dataclass
from enum import Enum
from typing import Optional


class BinaryRating(Enum):
    UPVOTE = 1
    NOVOTE = 0
    DOWNVOTE = -1


@dataclass
class Rating:
    rating: BinaryRating  # TODO: support other rating types (i.e. star scale, slider)
    feedback: Optional[str] = None
