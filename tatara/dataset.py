import logging
from typing import Optional, Dict, List, Any
from tatara.evals.id_generator import IdGenerator
from dataclasses import dataclass
from requests.models import Response
from requests.exceptions import HTTPError
from tatara.tatara import _get_network_client
from tatara.evals.record import Record

@dataclass
class Dataset:
    name: str
    records: List[Record]
    """
    You should never instantiate a Dataset directly. Instead, use the `init_dataset` function.
    """

    def __iter__(self):
        return iter(self.records)

    def __str__(self) -> str:
        return f"Dataset(name={self.name})\n\n{self.records.__str__()}"

    def __repr__(self) -> str:
        return f"Dataset(name={self.name})\n\n{self.records.__repr__()}"

    @property
    def size(self) -> int:
        return len(self.records)

    def _is_valid_record(self, record: Dict[str, Any]) -> bool:
        required_fields = {"input", "output"}
        for required_field in required_fields:
            if required_field not in record.keys():
                return False
        return True

    def insert(self, record_dicts: List[Dict[str, Any]]) -> Optional[Dict]:
        """
        Insert a record to a dataset that consists of input, output, and metadata
        """
        for record_dict in record_dicts:
            if not self._is_valid_record(record_dict):
                logging.warning(
                    f"Record {record_dict} is not valid. must have input and output fields"
                )

        resp = _get_network_client().send_insert_records_post_request(
            dataset_name=self.name, records=record_dicts
        )
        if resp and resp.ok:
            for record_dict in record_dicts:
                # construct a Record from each record_dict
                record = Record(
                    id=record_dict["id"],
                    input=record_dict["input"],
                    output=record_dict["output"],
                )
                self.records.append(record)

    def attach_records(self, record_ids: List[str]) -> None:
        # attach records to a dataset that already exist
        IdGenerator.validate_record_ids(record_ids)
        resp = _get_network_client().send_attach_records_post_request(
            dataset_name=self.name, record_ids=record_ids
        )
        if resp and resp.ok:
            # Add the records to the current dataset
            records = resp.json()
            self.records.extend(records)

    def attach_record(self, record_id: str) -> None:
        # attach a record to a dataset that already exists
        self.attach_records([record_id])


def init_dataset(name: str) -> Dataset:
    """
    Initialize a dataset. This will create a dataset on the tatara server asynchonously
    and return a new dataset object
    """
    _get_network_client().send_create_dataset_post_request(name)
    return Dataset(name=name, records=[])


def get_dataset(name: str) -> Dataset:  # type: ignore
    """
    Get a dataset from the tatara server by name
    """
    try:
        ds_data_response: Response = _get_network_client().send_dataset_get_request(
            name
        )
        ds_data = ds_data_response.json()
        return Dataset(name=name, records=ds_data["records"])
    except HTTPError as e:
        if e.response.status_code == 404:
            print(f'`get_dataset` failed. The dataset "{name}" does not exist.')
        else:
            raise
    except Exception as e:
        print(f"Exception occurred when getting dataset: {e}")
