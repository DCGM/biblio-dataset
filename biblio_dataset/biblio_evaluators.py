import logging
import os
from typing import Optional, Tuple, List

from pydantic import BaseModel
from abc import ABC, abstractmethod

from biblio_dataset.create_biblio_dataset import BiblioRecord, normalize_publisher, normalize_author, \
    normalize_illustrator, normalize_translator, normalize_editor, normalize_title, normalize_sub_title, \
    normalize_part_name, normalize_part_number, normalize_series_name, normalize_series_number, normalize_edition, \
    normalize_place_term, normalize_date_issued, normalize_manufacture_place_term, normalize_manufacture_publisher, \
    label_studio_classes_to_biblio_record_classes
from Levenshtein import distance

from detector_wrapper.parsers.pero_ocr import ALTOMatch

logger = logging.getLogger(__name__)

class BiblioResult(BaseModel):
    task_id: Optional[str] = None
    library_id: str

    title: Optional[List[Tuple[str, float]]] = None
    subTitle: Optional[List[Tuple[str, float]]] = None
    partName: Optional[List[Tuple[str, float]]] = None
    partNumber: Optional[List[Tuple[str, float]]] = None
    seriesName: Optional[List[Tuple[str, float]]] = None
    seriesNumber: Optional[List[Tuple[str, float]]] = None

    edition: Optional[List[Tuple[str, float]]] = None
    placeTerm: Optional[List[Tuple[str, float]]] = None
    dateIssued: Optional[List[Tuple[str, float]]] = None

    publisher: Optional[List[Tuple[str, float]]] = None
    manufacturePublisher: Optional[List[Tuple[str, float]]] = None
    manufacturePlaceTerm: Optional[List[Tuple[str, float]]] = None

    author: Optional[List[Tuple[str, float]]] = None
    illustrator: Optional[List[Tuple[str, float]]] = None
    translator: Optional[List[Tuple[str, float]]] = None
    editor: Optional[List[Tuple[str, float]]] = None


class BiblioStat(BaseModel):
    task_id: Optional[str] = None
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
    """
    Base class for Biblio Evaluators

    compare_list and compare_string methods must be implemented in the child class
        - use biblio_class if you want to evaluate each class differently,
          all evaluated classes can be found in EvalBiblioRecord
    """
    def __init__(self):
        self.biblio_record_stats: List[BiblioStat] = []
        self.biblio_result_true_positive_stats: List[BiblioStat] = []
        self.biblio_result_false_positive_stats: List[BiblioStat] = []

    def compare_biblio_record_to_result(self, biblio_record: BiblioRecord, biblio_result: BiblioResult):
        biblio_record_stat = {'task_id': biblio_record.task_id, 'library_id': biblio_record.library_id}
        biblio_record = biblio_record.model_dump(exclude_none=True)
        biblio_result_true_positive_stat = {'library_id': biblio_result.library_id}
        biblio_result_false_positive_stat = {'library_id': biblio_result.library_id}
        biblio_result = biblio_result.model_dump(exclude_none=True)

        for key, value in biblio_record.items():
            if key in ['task_id', 'library_id']:
                continue
            if isinstance(value, str):
                value = [value]
            biblio_record_stat[key] = len(value)
            if key in biblio_result:
                biblio_result_true_positive_stat[key] = self.compare_list(value, biblio_result[key], key)
                biblio_result_false_positive_stat[key] = min(len(biblio_result[key]), len(biblio_record[key])) - biblio_result_true_positive_stat[key]


        for key, value in biblio_result.items():
            if key in ['task_id', 'library_id']:
                continue
            if key not in biblio_record:
                biblio_result_false_positive_stat[key] = 1

        biblio_record_stat = BiblioStat.model_validate(biblio_record_stat)
        biblio_result_true_positive_stat = BiblioStat.model_validate(biblio_result_true_positive_stat)
        biblio_result_false_positive_stat = BiblioStat.model_validate(biblio_result_false_positive_stat)

        self.biblio_record_stats.append(biblio_record_stat)
        self.biblio_result_true_positive_stats.append(biblio_result_true_positive_stat)
        self.biblio_result_false_positive_stats.append(biblio_result_false_positive_stat)

    @abstractmethod
    def compare_list(self, record_list: List[str], result_list: List[Tuple[str, float]], biblio_class: str) -> int:
        pass

    @staticmethod
    def get_cer(record_string: str, result_string: str) -> float:
        return distance(record_string, result_string) / float(len(record_string))

    def get_stats(self):
        stats = {}
        for key in list(BiblioStat.model_fields.keys()):
            if key in ['task_id', 'library_id']:
                continue
            stats[key] = {
                'record': sum([getattr(record, key) for record in self.biblio_record_stats]),
                'result_true_positive': sum([getattr(record, key) for record in self.biblio_result_true_positive_stats]),
                'result_false_positive': sum([getattr(record, key) for record in self.biblio_result_false_positive_stats]),
            }
        return stats


class CERBiblioEvaluator(BaseBiblioEvaluator):
    def __init__(self, max_cer=0.1, top_k=1):
        super().__init__()
        self.max_cer = max_cer
        self.top_k = top_k

    def compare_list(self, record_list: List[str], result_list: List[Tuple[str, float]], biblio_class: str) -> int:
        hits = 0
        ignore_record_i = []
        result_list = sorted(result_list, key=lambda x: x[1], reverse=True)
        list_top_k = self.top_k * len(record_list)
        for k, (result_item, result_conf) in enumerate(result_list):
            if k >= list_top_k:
                break
            for record_i, record_item in enumerate(record_list):
                if record_i in ignore_record_i:
                    continue
                cer = self.get_cer(record_item, result_item)
                if cer <= self.max_cer:
                    hits += 1
                    ignore_record_i.append(record_i)
                    break
        return hits



def normalize_biblio_result(biblio_result: BiblioResult):
    normalized_biblio_result = BiblioResult(task_id=biblio_result.task_id,
                                            library_id=biblio_result.library_id)

    if biblio_result.title is not None:
        normalized_titles = []
        for title in biblio_result.title:
            normalized_titles.append((normalize_title(title[0]), title[1]))
        normalized_biblio_result.title = normalized_titles

    if biblio_result.subTitle is not None:
        normalized_sub_titles = []
        for sub_title in biblio_result.subTitle:
            normalized_sub_titles.append((normalize_sub_title(sub_title[0]), sub_title[1]))
        normalized_biblio_result.subTitle = normalized_sub_titles

    if biblio_result.partName is not None:
        normalized_part_names = []
        for part_name in biblio_result.partName:
            normalized_part_names.append((normalize_part_name(part_name[0]), part_name[1]))
        normalized_biblio_result.partName = normalized_part_names

    if biblio_result.partNumber is not None:
        normalized_part_numbers = []
        for part_number in biblio_result.partNumber:
            normalized_part_numbers.append((normalize_part_number(part_number[0]), part_number[1]))
        normalized_biblio_result.partNumber = normalized_part_numbers

    if biblio_result.seriesName is not None:
        normalized_series_names = []
        for series_name in biblio_result.seriesName:
            normalized_series_names.append((normalize_series_name(series_name[0]), series_name[1]))
        normalized_biblio_result.seriesName = normalized_series_names

    if biblio_result.seriesNumber is not None:
        normalized_series_numbers = []
        for series_number in biblio_result.seriesNumber:
            normalized_series_numbers.append((normalize_series_number(series_number[0]), series_number[1]))
        normalized_biblio_result.seriesNumber = normalized_series_numbers

    if biblio_result.edition is not None:
        normalized_editions = []
        for edition in biblio_result.edition:
            normalized_editions.append((normalize_edition(edition[0]), edition[1]))
        normalized_biblio_result.edition = normalized_editions

    if biblio_result.placeTerm is not None:
        normalized_place_terms = []
        for place_term in biblio_result.placeTerm:
            normalized_place_terms.append((normalize_place_term(place_term[0]), place_term[1]))
        normalized_biblio_result.placeTerm = normalized_place_terms

    if biblio_result.dateIssued is not None:
        normalized_date_issued = []
        for date_issued in biblio_result.dateIssued:
            normalized_date_issued.append((normalize_date_issued(date_issued[0]), date_issued[1]))
        normalized_biblio_result.dateIssued = normalized_date_issued

    if biblio_result.manufacturePublisher is not None:
        normalized_manufacture_publishers = []
        for manufacture_publisher in biblio_result.manufacturePublisher:
            normalized_manufacture_publishers.append((normalize_manufacture_publisher(manufacture_publisher[0]), manufacture_publisher[1]))
        normalized_biblio_result.manufacturePublisher = normalized_manufacture_publishers

    if biblio_result.manufacturePlaceTerm is not None:
        normalized_manufacture_place_terms = []
        for manufacture_place_term in biblio_result.manufacturePlaceTerm:
            normalized_manufacture_place_terms.append((normalize_manufacture_place_term(manufacture_place_term[0]), manufacture_place_term[1]))
        normalized_biblio_result.manufacturePlaceTerm = normalized_manufacture_place_terms

    if biblio_result.publisher is not None:
        normalized_publishers = []
        for publisher in biblio_result.publisher:
            normalized_publishers.append((normalize_publisher(publisher[0]), publisher[1]))
        normalized_biblio_result.publisher = normalized_publishers

    if biblio_result.author is not None:
        normalized_authors = []
        for author in biblio_result.author:
            normalized_authors.append((normalize_author(author[0]), author[1]))
        normalized_biblio_result.author = normalized_authors

    if biblio_result.illustrator is not None:
        normalized_illustrators = []
        for illustrator in biblio_result.illustrator:
            normalized_illustrators.append((normalize_illustrator(illustrator[0]), illustrator[1]))
        normalized_biblio_result.illustrator = normalized_illustrators

    if biblio_result.translator is not None:
        normalized_translators = []
        for translator in biblio_result.translator:
            normalized_translators.append((normalize_translator(translator[0]), translator[1]))
        normalized_biblio_result.translator = normalized_translators

    if biblio_result.editor is not None:
        normalized_editors = []
        for editor in biblio_result.editor:
            normalized_editors.append((normalize_editor(editor[0]), editor[1]))
        normalized_biblio_result.editor = normalized_editors

    return normalized_biblio_result


def convert_alto_match_to_biblio_results(alto_match: ALTOMatch):
    biblio_results = []
    logger.info(f"Converting {len(alto_match.matched_pages)} matched pages to BiblioResults")
    for matched_page in alto_match.matched_pages:
        biblio_result = {'task_id': matched_page.detector_parser_page.id,
                         'library_id': os.path.splitext(matched_page.detector_parser_page.image_filename)[0]}
        for matched_detection in matched_page.matched_detections:
            detection_class = matched_detection.get_class()
            detection_value = matched_detection.get_text()
            detection_confidence = matched_detection.get_confidence()

            if detection_confidence is None:
                logger.warning(f"Detection confidence is None, "
                               f"detector parser page image filename: {matched_page.detector_parser_page.image_filename}, "
                               f"detector parser page bbox : {matched_detection.detector_parser_annotated_bounding_box} -> "
                               f"setting detection confidence to 0.0")
                detection_confidence = 0.0
            if detection_class not in label_studio_classes_to_biblio_record_classes:
                logger.warning(f"{detection_class} class invalid, "
                               f"detector parser image filename: {matched_page.detector_parser_page.id}, "
                               f"detector parser page bbox : {matched_detection.detector_parser_annotated_bounding_box} -> "
                               f"skipping detection")
                continue
            biblio_record_class = label_studio_classes_to_biblio_record_classes[detection_class]
            if biblio_result.get(biblio_record_class) is None:
                biblio_result[biblio_record_class] = []
            biblio_result[biblio_record_class].append((detection_value, detection_confidence))

        biblio_result = BiblioResult.model_validate(biblio_result)
        biblio_results.append(biblio_result)
    logger.info(f"Converted {len(biblio_results)} matched pages to BiblioResults")
    return biblio_results