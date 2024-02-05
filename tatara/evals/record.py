from dataclasses import dataclass, field
from typing import Optional

from tatara.evals.model_package import ModelPackage


@dataclass
class Record:
    id: str = field(init=False)
    input: str
    output: str
    model_package: Optional[ModelPackage] = None

    def __post_init__(self):
        # check that input and output are not None
        if not self.input:
            raise ValueError("input cannot be None")
        if not self.output:
            raise ValueError("output cannot be None")

    @property
    def model_input_type(self):
        return self.model_package.model_input_type if self.model_package else None

    @property
    def model_output_type(self):
        return self.model_package.model_output_type if self.model_package else None

    def to_dict(self):
        return {
            "id": self.id,
            "input": self.input,
            "output": self.output,
            "model_package": self.model_package.to_dict()
            if self.model_package
            else None,
        }
