import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Union


class ModelInputType(Enum):
    TEXT = "text"
    IMAGE = "image"


class ModelOutputType(Enum):
    TEXT = "text"
    IMAGE = "image"


@dataclass
class ModelPackage:
    model_name: str
    hyperparameters: Dict[
        str, Union[str, float, int]
    ]  # this includes all of the hyperparameters for the model including the system prompt if there is one
    model_input_type: ModelInputType
    model_output_type: ModelOutputType
    model_sha256: str = field(
        init=False
    )  # a sha256 that uniquely identifies the model and hyperparamters

    @staticmethod
    def compute_model_sha256(
        model_name: str, hyperparameters: Dict[str, Union[str, float, int]]
    ) -> str:
        sorted_hyperparameters: str = json.dumps(hyperparameters, sort_keys=True)
        model_hash = hashlib.sha256(
            (model_name + sorted_hyperparameters).encode()
        ).hexdigest()
        return f"modelsha256_{model_hash}"

    def __post_init__(self):
        self.model_hash = ModelPackage.compute_model_sha256(
            self.model_name, self.hyperparameters
        )

    def __str__(self):
        return f"ModelPackage(model_name={self.model_name}, model_hash={self.model_hash[:20]}..., input_type={self.model_input_type}, output_type={self.model_output_type})"

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "hyperparameters": self.hyperparameters,
            "model_input_type": self.model_input_type.value,
            "model_output_type": self.model_output_type.value,
            "model_sha256": self.model_sha256,
        }
