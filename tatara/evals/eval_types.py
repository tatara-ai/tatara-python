from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from tatara.evals.id_generator import IdGenerator

RecordId = str

EvalResult = Union[bool, int, float, str]

@dataclass
class EvalResultWithMetadata:
    eval_name: str
    eval_description: str
    result: EvalResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eval_name": self.eval_name,
            "eval_description": self.eval_description,
            "result": self.result,
        }


@dataclass
class RecordWithSingleEvalResult:
    record_id: RecordId
    input: str
    output: str
    eval_result_with_metadata: EvalResultWithMetadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "input": self.input,
            "output": self.output,
            "eval_result_with_metadata": self.eval_result_with_metadata.__dict__,
        }


@dataclass
class RecordWithMultipleEvalResults:
    id: str = field(init=False)
    record_id: RecordId
    input: str
    output: str
    eval_results_with_metadata: List[EvalResultWithMetadata]

    def __post_init__(self):
        self.id = IdGenerator.generate_eval_run_row_uuid()

    def add(self, record_with_single_eval: RecordWithSingleEvalResult) -> None:
        self.eval_results_with_metadata.append(record_with_single_eval.eval_result_with_metadata)
    
    @property
    def num_evals(self) -> int:
        return len(self.eval_results_with_metadata)

    def __iter__(self):
        return iter(self.eval_results_with_metadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "record_id": self.record_id,
            "input": self.input,
            "output": self.output,
            "eval_results_with_metadata": [record_with_single_eval.__dict__ for record_with_single_eval in self.eval_results_with_metadata],
        }


@dataclass
class EvalRun:
    id: str = field(init=False)
    eval_rows: List[RecordWithMultipleEvalResults]

    @property
    def num_rows(self) -> int:
        return len(self.eval_rows)

    @property
    def num_evals(self) -> int:
        return sum([eval_row.num_evals for eval_row in self.eval_rows])

    def __post_init__(self):
        self.id = IdGenerator.generate_eval_run_uuid()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "eval_rows": [eval_row.to_dict() for eval_row in self.eval_rows],
        }
