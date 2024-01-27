from typing import Optional
import re


def _gen_id_from_trace_and_event(trace_id: str, event: Optional[str]) -> str:
    if event:
        return event + "_" + trace_id
    return trace_id


def _check_event(event: str) -> None:
    if len(event) > 64:
        raise ValueError(
            f"'{event}' is too long of a event. Must be under 64 characters."
        )
    if not re.match(r"^\w+$", event):
        raise ValueError(
            f"'{event}' is an invalid event. Must consist of only alphanumeric characters and underscores."
        )
