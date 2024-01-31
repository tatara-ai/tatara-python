from dataclasses import dataclass, field
from typing import Any, Dict, List

from tatara.evals.id_generator import IdGenerator

RecordId = str


@dataclass
class EvalRecord:
    eval_name: str
    eval_description: str
    result: bool | None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eval_name": self.eval_name,
            "eval_description": self.eval_description,
            "result": self.result,
        }


@dataclass
class EvalValue:
    record_id: RecordId
    input: str
    output: str
    eval_record: EvalRecord

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "input": self.input,
            "output": self.output,
            "eval_record": self.eval_record.__dict__,
        }


@dataclass
class EvalRow:
    id: str = field(init=False)
    record_id: RecordId
    input: str
    output: str
    eval_values: List[EvalRecord]

    def __post_init__(self):
        self.id = IdGenerator.generate_eval_run_row_uuid()

    def add(self, eval_value: EvalValue) -> None:
        self.eval_values.append(eval_value.eval_record)

    def __iter__(self):
        return iter(self.eval_values)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "record_id": self.record_id,
            "input": self.input,
            "output": self.output,
            "eval_values": [eval_value.__dict__ for eval_value in self.eval_values],
        }


@dataclass
class EvalRun:
    id: str = field(init=False)
    eval_rows: List[EvalRow]

    def __post_init__(self):
        self.id = IdGenerator.generate_eval_run_uuid()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "eval_rows": [eval_row.to_dict() for eval_row in self.eval_rows],
        }
