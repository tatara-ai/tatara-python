import uuid
from typing import List


class IdPrefix:
    RECORD = "r_"
    DATASET = "d_"
    EVAL_RUN = "er_"
    EVAL_RUN_ROW = "err_"


class IdGenerator:
    @staticmethod
    def generate_record_uuid() -> str:
        return IdPrefix.RECORD + str(uuid.uuid4())

    @staticmethod
    def generate_dataset_uuid() -> str:
        return IdPrefix.DATASET + str(uuid.uuid4())

    @staticmethod
    def generate_eval_run_row_uuid() -> str:
        return IdPrefix.EVAL_RUN_ROW + str(uuid.uuid4())

    @staticmethod
    def generate_eval_run_uuid() -> str:
        return IdPrefix.EVAL_RUN + str(uuid.uuid4())

    @staticmethod
    def is_record_id(id: str) -> bool:
        return id.startswith(IdPrefix.RECORD)

    @staticmethod
    def validate_record_ids(ids: List[str]) -> None:
        for id in ids:
            if not IdGenerator.is_record_id(id):
                raise ValueError(f"Invalid record id: {id}")
