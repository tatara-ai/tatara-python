from dataclasses import dataclass
from typing import Optional, Dict, Any



@dataclass
class Record:
    id: str
    input: str
    output: str
    params: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # check that id, input, and output are not None
        if not self.id:
            raise ValueError("id cannot be None")
        if not self.input:
            raise ValueError("input cannot be None")
        if not self.output:
            raise ValueError("output cannot be None")
        # TODO: consider creating a model package from the params

    def to_dict(self):
        return {
            "id": self.id,
            "input": self.input,
            "output": self.output,
            "params": self.params,
        }
