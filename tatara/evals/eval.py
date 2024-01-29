import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from evals.eval_types import EvalRecord, EvalRow, EvalRun, EvalValue
from evals.model_package import ModelInputType, ModelOutputType
from evals.record import Record
from evals.recorder import RecorderBase


class Eval(ABC):
    name: str
    description: str
    valid_input_types: List[ModelInputType]
    valid_output_types: List[ModelOutputType]

    def __init__(
        self,
        records: list,
        seed: Optional[int] = 42,
    ):
        self.seed = seed
        self.records = records

    @abstractmethod
    def eval_record(self, record: Record) -> Optional[bool]:
        """
        Eval a single record
        """
        raise NotImplementedError

    def run_no_recorder(self) -> List[EvalValue]:
        """
        Run the eval without recording the results anywhere
        """
        eval_results = []
        for record in self.records:
            eval_result = EvalValue(
                record_id=record.record_id,
                input=record.input,
                output=record.output,
                eval_record=EvalRecord(
                    eval_name=self.name,
                    eval_description=self.description,
                    result=self.eval_record(record),
                ),
            )
            eval_results.append(eval_result)

        return eval_results

    def run(self, recorder: RecorderBase) -> None:
        """
        Run the eval with the RecorderBase
        """
        for record in self.records:
            eval_row = {
                "record_id": record.record_id,
                "input": record.input,
                "output": record.output,
                "eval": {
                    "name": self.name,
                    "description": self.description,
                    "result": self.eval_record(record),
                },
            }
            recorder.record_eval_row(eval_row)

    def is_valid_input_type(self, model_input_type: ModelInputType) -> bool:
        return model_input_type in self.valid_input_types

    def is_valid_output_type(self, model_output_type: ModelOutputType) -> bool:
        logging.warning("")
        return model_output_type in self.valid_output_types


@dataclass
class Evals:
    evals: List[Eval]

    def eval_values_to_rows(
        self, all_eval_results: List[List[EvalValue]]
    ) -> List[EvalRow]:
        eval_rows = {}
        for eval_results in all_eval_results:
            for eval_result in eval_results:
                # join the eval result for the record into a single row
                record_id = eval_result.record_id
                if record_id in eval_rows:
                    # just add the eval result to the existing record in the output
                    eval_rows[record_id].add(eval_result)
                else:
                    eval_rows[record_id] = EvalRow(
                        record_id=record_id,
                        input=eval_result.input,
                        output=eval_result.output,
                        eval_values=[eval_result.eval_record],
                    )
        return list(eval_rows.values())

    def run(self, recorder: RecorderBase):
        all_eval_results = []
        for eval in self.evals:
            eval_results: List[EvalValue] = eval.run_no_recorder()
            all_eval_results.append(eval_results)

        eval_rows = self.eval_values_to_rows(all_eval_results)
        eval_run = EvalRun(eval_rows=eval_rows)
        # TODO: record the eval run as well via a separate endpoint in the recorder
        recorder.record_eval_run(eval_run)
