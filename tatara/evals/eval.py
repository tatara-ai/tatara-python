from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from tatara.evals.eval_types import EvalResultWithMetadata, RecordWithEvalResults, EvalResult
from tatara.evals.record import Record
from tatara.evals.recorder import RecorderBase
from enum import Enum
from typing import Any, Dict
from dataclasses import field
from tatara.evals.id_generator import IdGenerator
from tatara.dataset import Dataset

class EvalResultType(Enum):
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    CATEGORICAL = "categorical"

    def __str__(self):
        return self.value
    
    def to_str(self):
        return self.value


class Eval(ABC):
    name: str
    description: str
    eval_result_type: EvalResultType

    def __init__(
        self,
        records: list,
        dataset: Optional[Dataset] = None,
        seed: Optional[int] = 42,
    ):
        self.seed = seed
        self.records = records
        self.dataset = dataset
    

    @abstractmethod
    def eval_record(self, record: Record) -> EvalResult:
        """
        Eval a single record
        """
        raise NotImplementedError

    def run_no_recorder(self) -> "EvalRun":
        """
        Run the eval without recording the results anywhere
        """
        single_eval_all_records = []
        for record in self.records:
            record_with_single_eval_result = RecordWithEvalResults(
                record_id=record.id,
                input=record.input,
                output=record.output,
                eval_results_with_metadata=[EvalResultWithMetadata(
                    eval_name=self.name,
                    eval_description=self.description,
                    result=self.eval_record(record),
                )],
            )
            single_eval_all_records.append(record_with_single_eval_result)

        return EvalRun(eval_rows=single_eval_all_records, evals=[self])

    def run(self, recorder: RecorderBase) -> None:
        """
        Run the eval with the RecorderBase
        """
        single_eval_all_records = self.run_no_recorder()
        recorder.record_eval_run(single_eval_all_records)
        
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "eval_result_type": self.eval_result_type.to_str(),
        }

@dataclass
class Evals:
    evals: List[Eval]

    def combine_single_eval_results_to_row(
        self, all_eval_runs: List["EvalRun"]
    ) -> List[RecordWithEvalResults]:
        record_with_multiple_eval_results = {}
        for eval in all_eval_runs:
            for record_with_single_eval_result in eval.eval_rows:
                # join the eval result for the record into a single row
                record_id = record_with_single_eval_result.record_id
                if record_id in record_with_multiple_eval_results:
                    # just add the eval result to the existing record in the output
                    record_with_multiple_eval_results[record_id].add(record_with_single_eval_result)
                else:
                    record_with_multiple_eval_results[record_id] = RecordWithEvalResults(
                        record_id=record_id,
                        input=record_with_single_eval_result.input,
                        output=record_with_single_eval_result.output,
                        eval_results_with_metadata=record_with_single_eval_result.eval_results_with_metadata,
                    )
        return list(record_with_multiple_eval_results.values())

    def run(self, recorder: RecorderBase):
        all_eval_runs = []
        for eval in self.evals:
            eval_run: EvalRun = eval.run_no_recorder()
            all_eval_runs.append(eval_run)
        eval_rows = self.combine_single_eval_results_to_row(all_eval_runs)
        eval_run = EvalRun(evals=self.evals, eval_rows=eval_rows)
        recorder.record_eval_run(eval_run)



@dataclass
class EvalRun:
    id: str = field(init=False)
    evals: List[Eval]
    eval_rows: List[RecordWithEvalResults]
    dataset: Optional[Dataset] = None
    
    @property
    def num_rows(self) -> int:
        return len(self.eval_rows)

    @property
    def num_evals(self) -> int:
        return sum([eval_row.num_evals for eval_row in self.eval_rows])

    def __post_init__(self):
        self.id = IdGenerator.generate_eval_run_uuid()
        # set the run id on the rows
        for eval_row in self.eval_rows:
            eval_row.eval_run_id = self.id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "dataset": {"id": self.dataset.id, "name": self.dataset.name} if self.dataset else None,
            "evals": [eval.to_dict() for eval in self.evals],
            "eval_run_id": self.id,
            "eval_rows": [eval_row.to_dict() for eval_row in self.eval_rows],
        }
