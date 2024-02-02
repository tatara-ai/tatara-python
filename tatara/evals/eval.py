import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from tatara.evals.eval_types import EvalResultWithMetadata, RecordWithMultipleEvalResults, EvalRun, EvalResult, RecordWithSingleEvalResult
from tatara.evals.model_package import ModelInputType, ModelOutputType
from tatara.evals.record import Record
from tatara.evals.recorder import RecorderBase
from enum import Enum

class EvalResultType(Enum):
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    CATEGORICAL = "categorical"


class Eval(ABC):
    name: str
    description: str
    valid_input_types: List[ModelInputType]
    valid_output_types: List[ModelOutputType]
    eval_result_type: EvalResultType

    def __init__(
        self,
        records: list,
        seed: Optional[int] = 42,
    ):
        self.seed = seed
        self.records = records

    @abstractmethod
    def eval_record(self, record: Record) -> EvalResult:
        """
        Eval a single record
        """
        raise NotImplementedError

    def run_no_recorder(self) -> EvalRun:
        """
        Run the eval without recording the results anywhere
        """
        single_eval_all_records = []
        for record in self.records:
            record_with_single_eval_result = RecordWithSingleEvalResult(
                record_id=record.id,
                input=record.input,
                output=record.output,
                eval_result_with_metadata=EvalResultWithMetadata(
                    eval_name=self.name,
                    eval_description=self.description,
                    result=self.eval_record(record),
                ),
            )
            single_eval_all_records.append(record_with_single_eval_result)

        return EvalRun(eval_rows=single_eval_all_records)

    def run(self, recorder: RecorderBase) -> None:
        """
        Run the eval with the RecorderBase
        """
        single_eval_all_records = self.run_no_recorder()
        recorder.record_eval_run(single_eval_all_records)
        
    
    def is_valid_input_type(self, model_input_type: ModelInputType) -> bool:
        return model_input_type in self.valid_input_types

    def is_valid_output_type(self, model_output_type: ModelOutputType) -> bool:
        logging.warning("")
        return model_output_type in self.valid_output_types


@dataclass
class Evals:
    evals: List[Eval]

    def eval_values_to_rows(
        self, all_evals_all_records: List[List[RecordWithSingleEvalResult]]
    ) -> List[RecordWithMultipleEvalResults]:
        record_with_multiple_eval_results = {}
        for single_eval_all_records in all_evals_all_records:
            for record_with_single_eval_result in single_eval_all_records:
                # join the eval result for the record into a single row
                record_id = record_with_single_eval_result.record_id
                if record_id in record_with_multiple_eval_results:
                    # just add the eval result to the existing record in the output
                    record_with_multiple_eval_results[record_id].add(record_with_single_eval_result)
                else:
                    record_with_multiple_eval_results[record_id] = RecordWithMultipleEvalResults(
                        record_id=record_id,
                        input=record_with_single_eval_result.input,
                        output=record_with_single_eval_result.output,
                        eval_results_with_metadata=[record_with_single_eval_result.eval_result_with_metadata],
                    )
        return list(record_with_multiple_eval_results.values())

    def run(self, recorder: RecorderBase):
        all_evals_all_records = []
        for eval in self.evals:
            single_eval_all_records: EvalRun = eval.run_no_recorder()
            all_evals_all_records.append(single_eval_all_records)

        eval_rows = self.eval_values_to_rows(all_evals_all_records)
        eval_run = EvalRun(eval_rows=eval_rows)
        recorder.record_eval_run(eval_run)
