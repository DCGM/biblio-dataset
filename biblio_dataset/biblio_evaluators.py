from pydantic import BaseModel
from abc import ABC, abstractmethod

from biblio_dataset.create_biblio_dataset import BiblioRecord

class EvalBiblioRecord(BaseModel):
    task_id: int
    library_id: str

    title: int = 0
    subTitle: int = 0
    partName: int = 0
    partNumber: int = 0
    seriesName: int = 0
    seriesNumber: int = 0

    edition: int = 0
    placeTerm: int = 0
    dateIssued: int = 0

    publisher: int = 0
    manufacturePublisher: int = 0
    manufacturePlaceTerm: int = 0

    author: int = 0
    illustrator: int = 0
    translator: int = 0
    editor: int = 0

class BaseBiblioEvaluator(ABC):
    def __init__(self):
        self.gt_eval_biblio_records = []
        self.result_eval_biblio_records = []

    def compare_biblio_records(self, gt_record: BiblioRecord, result_record: BiblioRecord):
        gt_eval_biblio_record = {'id': gt_record.task_id, 'library_id': gt_record.library_id}
        gt_record = gt_record.model_dump(exclude_none=True)
        result_eval_biblio_record = {'id': result_record.task_id, 'library_id': result_record.library_id}
        result_record = result_record.model_dump(exclude_none=True)

        for key, value in gt_record.items():
            if isinstance(value, list):
                gt_eval_biblio_record[key] = len(value)
                if key in result_record:
                    result_eval_biblio_record[key] = self.compare_list(value, result_record[key], key)
            elif value is not None:
                gt_eval_biblio_record[key] = 1
                if key in result_record:
                    result_eval_biblio_record[key] = self.compare_string(value, result_record[key], key)

        gt_eval_biblio_record = EvalBiblioRecord.model_validate(gt_eval_biblio_record)
        result_eval_biblio_record = EvalBiblioRecord.model_validate(result_eval_biblio_record)

        self.gt_eval_biblio_records.append(gt_eval_biblio_record)
        self.result_eval_biblio_records.append(result_eval_biblio_record)

    @abstractmethod
    def compare_list(self, gt_list, result_list, biblio_class: str) -> int:
        pass

    @abstractmethod
    def compare_string(self, gt_string, result_string, biblio_class: str) -> int:
        pass


class CERBiblioEvaluator(BaseBiblioEvaluator):
    def __init__(self, max_cer=0.0):
        super().__init__()
        self.max_cer = max_cer

    def compare_list(self, gt_list, result_list, biblio_class: str) -> int:
        return len(gt_list) == len(result_list)

    def compare_string(self, gt_string, result_string, biblio_class: str) -> int:
        return int(gt_string == result_string)