from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional
from tatara.evals.eval_types import EvalResultWithMetadata, RecordWithEvalResults, EvalResult
from tatara.evals.record import Record
from tatara.evals.recorder import PrintRecorder, RecorderBase
from tatara.evals.tatara_recorder import TataraRecorder
from typing import Any, Dict
from dataclasses import field
from tatara.evals.id_generator import IdGenerator
import typing
import time


@dataclass
class Eval:
    name: str
    eval_fn: Callable[[Record], EvalResult]
    description: Optional[str] = None
    seed: Optional[int] = 42


    def run_no_recorder(self, dataset: Any) -> "EvalRun": # TODO: make this a dataset type
        """
        Run the eval without recording the results anywhere
        """
        single_eval_all_records = []
        for record in dataset.records:
            record_with_single_eval_result = RecordWithEvalResults(
                record_id=record['id'],
                input=record['input'],
                output=record['output'],
                eval_results_with_metadata=[EvalResultWithMetadata(
                    eval_name=self.name,
                    eval_description=self.description if self.description else "",
                    result=self.eval_fn(record),
                )],
            )
            single_eval_all_records.append(record_with_single_eval_result)

        return EvalRun(eval_rows=single_eval_all_records, evals=[self], dataset=dataset)

    def run(self, recorder: RecorderBase, dataset: Any) -> None: # TODO: make this a dataset type
        """
        Run the eval with the RecorderBase
        """
        single_eval_all_records = self.run_no_recorder(dataset=dataset)
        recorder.record_eval_run(single_eval_all_records)
        
    
    def to_dict(self) -> Dict[str, Any]:
        return_type_raw_str = typing.get_type_hints(self.eval_fn).get("return")
        eval_result_type = "unknown"
        if return_type_raw_str == float:
            eval_result_type = "float"
        elif return_type_raw_str == int:
            eval_result_type = "int"
        elif return_type_raw_str == str:
            eval_result_type = "str"
        elif return_type_raw_str == bool:
            eval_result_type = "bool"
        return {
            "name": self.name,
            "description": self.description,
            "eval_result_type": eval_result_type,
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

    def run(self, dataset: Any, print_only: bool = False) -> None: # TODO: make this a dataset type
        """
        Run evals. If print_only is True, the evals will be printed to the console and not recorded.
        """
        recorder = TataraRecorder() if not print_only else PrintRecorder()
        
        all_eval_runs = []
        for eval in self.evals:
            eval_run: EvalRun = eval.run_no_recorder(dataset=dataset)
            all_eval_runs.append(eval_run)
        eval_rows = self.combine_single_eval_results_to_row(all_eval_runs)
        eval_run = EvalRun(evals=self.evals, eval_rows=eval_rows, dataset=dataset)
        recorder.record_eval_run(eval_run)



@dataclass
class EvalRun:
    id: str = field(init=False)
    evals: List[Eval]
    eval_rows: List[RecordWithEvalResults]
    dataset: Any # TODO: make this a dataset type
    timestamp: int = field(default_factory=lambda: int(time.time()))
    
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
            "num_rows": self.num_rows,
            "eval_rows": [eval_row.to_dict() for eval_row in self.eval_rows],
            "timestamp": self.timestamp,
        }
