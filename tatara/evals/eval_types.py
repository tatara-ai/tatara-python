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
class RecordWithEvalResults:
    id: str = field(init=False)
    record_id: RecordId
    eval_run_id: str = field(init=False) # gets set when we add rows to an eval run
    input: str
    output: str
    eval_results_with_metadata: List[EvalResultWithMetadata]

    def __post_init__(self):
        self.id = IdGenerator.generate_eval_run_row_uuid()

    def add(self, record_with_eval_results: "RecordWithEvalResults") -> None:
        self.eval_results_with_metadata.extend(record_with_eval_results.eval_results_with_metadata)
    
    @property
    def num_evals(self) -> int:
        return len(self.eval_results_with_metadata)

    def __iter__(self):
        return iter(self.eval_results_with_metadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "eval_run_id": self.eval_run_id,
            "record_id": self.record_id,
            "input": self.input,
            "output": self.output,
            "eval_results": [{record_with_single_eval.eval_name: record_with_single_eval.result} for record_with_single_eval in self.eval_results_with_metadata],
        }
