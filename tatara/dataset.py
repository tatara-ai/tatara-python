import logging
from typing import Optional, Dict, List, Any
from tatara.evals.id_generator import IdGenerator
from dataclasses import dataclass
from requests.models import Response
from requests.exceptions import HTTPError
from tatara.tatara import _get_network_client
from tatara.evals.eval import Evals, Eval


# TODO: make sure methods in Dataset are lazy and don't do anything until an op is called that needs the data
@dataclass
class Dataset:
    name: str
    records: List[Dict[str, Any]]
    id: Optional[str] = None
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

    def insert(self, records: List[Dict[str, Any]]) -> Optional[Dict]:
        """
        Insert a record to a dataset that consists of input, output, and metadata
        """
        for record in records:
            if not self._is_valid_record(record):
                logging.warning(
                    f"Record {record} is not valid. must have input and output fields"
                )

        resp = _get_network_client().send_insert_records_post_request(
            dataset_name=self.name, records=records
        )
        if resp and resp.ok:
            records_with_ids = resp.json()
            # Add the records to the current dataset
            self.records.extend(records_with_ids)

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
    
    def run_evals(self, list_of_evals: List[Eval], print_only: bool = False) -> None:
        """
        Run evals on the dataset. If local is True, the evals will be printed to the console.
        """
        evals = Evals(list_of_evals)
        evals.run(self, print_only=print_only)


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
        return Dataset(name=name, id=ds_data['id'], records=ds_data["records"])
    except HTTPError as e:
        if e.response.status_code == 404:
            print(f'`get_dataset` failed. The dataset "{name}" does not exist.')
        else:
            raise
    except Exception as e:
        print(f"Exception occurred when getting dataset: {e}")

def get_or_init_dataset(name: str) -> Dataset:
    """
    Get a dataset from the tatara server by name. If the dataset does not exist, it will be created.
    """
    try:
        return get_dataset(name)
    except Exception:
        return init_dataset(name)


def run_evals(list_of_evals: List[Eval], dataset_name: Any, print_only: bool = False) -> None: # TODO: make this a dataset type
    """
    Run evals on the dataset. If local is True, the evals will be printed to the console.
    """
    evals = Evals(list_of_evals)
    print("Fetching dataset: {}".format(dataset_name))
    dataset = get_dataset(dataset_name)
    print("Running evals on dataset.")
    evals.run(dataset, print_only=print_only)