import logging
import os
from typing import Optional, Tuple, List

import numpy as np
from pydantic import BaseModel
from abc import ABC, abstractmethod

from biblio_dataset.create_biblio_dataset import BiblioRecord, label_studio_classes_to_biblio_record_classes
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
    publisher: int = 0
    placeTerm: int = 0
    dateIssued: int = 0

    manufacturePublisher: int = 0
    manufacturePlaceTerm: int = 0

    author: int = 0
    illustrator: int = 0
    translator: int = 0
    editor: int = 0

class BiblioLabelConf(BaseModel):
    task_id: Optional[str] = None
    library_id: str

    title: Tuple[List[int], List[float]] = ([], [])
    subTitle: Tuple[List[int], List[float]] = ([], [])
    partName: Tuple[List[int], List[float]] = ([], [])
    partNumber: Tuple[List[int], List[float]] = ([], [])
    seriesName: Tuple[List[int], List[float]] = ([], [])
    seriesNumber: Tuple[List[int], List[float]] = ([], [])

    edition: Tuple[List[int], List[float]] = ([], [])
    placeTerm: Tuple[List[int], List[float]] = ([], [])
    dateIssued: Tuple[List[int], List[float]] = ([], [])

    publisher: Tuple[List[int], List[float]] = ([], [])
    manufacturePublisher: Tuple[List[int], List[float]] = ([], [])
    manufacturePlaceTerm: Tuple[List[int], List[float]] = ([], [])

    author: Tuple[List[int], List[float]] = ([], [])
    illustrator: Tuple[List[int], List[float]] = ([], [])
    translator: Tuple[List[int], List[float]] = ([], [])
    editor: Tuple[List[int], List[float]] = ([], [])


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
        self.biblio_result_label_conf: List[BiblioLabelConf] = []

    def compare_biblio_record_to_result(self, biblio_record: BiblioRecord, biblio_result: BiblioResult):
        biblio_record_stat = {'task_id': biblio_record.task_id, 'library_id': biblio_record.library_id}
        biblio_record = biblio_record.model_dump(exclude_none=False)
        biblio_result_true_positive_stat = {'library_id': biblio_result.library_id}
        biblio_result_false_positive_stat = {'library_id': biblio_result.library_id}
        biblio_result_label_conf = {'library_id': biblio_result.library_id}
        biblio_result = biblio_result.model_dump(exclude_none=True)

        for key, value in biblio_record.items():
            if key in ['task_id', 'library_id']:
                continue
            # FP
            if value is None and key in biblio_result and biblio_result[key] is not None:
                biblio_result_false_positive_stat[key] = len(biblio_result[key])
                labels = [0] * len(biblio_result[key])
                conf = [conf for _, conf in biblio_result[key]]
                biblio_result_label_conf[key] = (labels, conf)
                continue
            if value is None:
                continue
            if isinstance(value, str):
                value = [value]
            biblio_record_stat[key] = len(value)
            if key in biblio_result:
                tp, labels, conf = self.compare_list(value, biblio_result[key], key)
                biblio_result_true_positive_stat[key] = tp
                biblio_result_false_positive_stat[key] = len(biblio_result[key]) - tp
                if biblio_result_false_positive_stat[key] < 0:
                    logger.warning(f"False positive count is negative, this should not happen, fix compare_list to return correct count")
                biblio_result_label_conf[key] = (labels, conf)
            # FN
            else:
                labels = [-1] * len(value)
                conf = [0.0] * len(value)
                biblio_result_label_conf[key] = (labels, conf)


        biblio_record_stat = BiblioStat.model_validate(biblio_record_stat)
        biblio_result_true_positive_stat = BiblioStat.model_validate(biblio_result_true_positive_stat)
        biblio_result_false_positive_stat = BiblioStat.model_validate(biblio_result_false_positive_stat)
        biblio_result_label_conf = BiblioLabelConf.model_validate(biblio_result_label_conf)

        self.biblio_record_stats.append(biblio_record_stat)
        self.biblio_result_true_positive_stats.append(biblio_result_true_positive_stat)
        self.biblio_result_false_positive_stats.append(biblio_result_false_positive_stat)
        self.biblio_result_label_conf.append(biblio_result_label_conf)

    @abstractmethod
    def compare_list(self, record_list: List[str], result_list: List[Tuple[str, float]], biblio_class: str) -> Tuple[int, List[int], List[float]]:
        pass

    @staticmethod
    def get_cer(record_string: str, result_string: str) -> float:
        return distance(record_string, result_string) / float(len(record_string))

    @staticmethod
    def get_substr_cer(record_string: str, result_string: str, sub_cost=1, ins_cost=1, del_cost=1) -> float:
        target = list(record_string)
        source = list(result_string)
        if len(target) > len(source):
            raise ValueError("Record string must be shorter than result string")

        target = np.array(target)
        dist = np.ones((1 + len(target) + 1)) * float('inf')
        dist[0] = 0
        for s in source:
            dist[1:-1] = np.minimum(dist[1:-1] + del_cost, dist[:-2] + (target != s) * sub_cost)

            for ii in range(len(dist) - 2):
                if dist[ii + 1] > dist[ii] + ins_cost:
                    dist[ii + 1] = dist[ii] + ins_cost

            dist[-1] = np.minimum(dist[-1], dist[-2])

        return dist[-1]

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
            stats[key]['result_label'] = []
            stats[key]['result_conf'] = []
            for record in self.biblio_result_label_conf:
                labels, conf = getattr(record, key)
                stats[key]['result_label'] += labels
                stats[key]['result_conf'] += conf
        return stats


class CERBiblioEvaluator(BaseBiblioEvaluator):
    def __init__(self, max_cer=0.1):
        super().__init__()
        self.max_cer = max_cer

    def compare_list(self, record_list: List[str], result_list: List[Tuple[str, float]], biblio_class: str) -> Tuple[int, List[int], List[float]]:
        hits = 0
        labels = []
        conf = []
        ignore_record_i = []
        result_list = sorted(result_list, key=lambda x: x[1], reverse=True)
        for k, (result_item, result_conf) in enumerate(result_list):
            record_found = False
            for record_i, record_item in enumerate(record_list):
                if record_i in ignore_record_i:
                    continue
                cer = self.get_cer(record_item, result_item)
                # TP
                if cer <= self.max_cer:
                    hits += 1
                    labels.append(1)
                    conf.append(result_conf)
                    ignore_record_i.append(record_i)
                    record_found = True
                    break
            # FP
            if not record_found:
                labels.append(0)
                conf.append(result_conf)
        # FN
        for record_i, record_item in enumerate(record_list):
            if record_i in ignore_record_i:
                continue
            labels.append(-1)
            conf.append(0.0)

        return hits, labels, conf


class CERBiblioSubstringEvaluator(BaseBiblioEvaluator):
    def __init__(self, max_cer=0.1):
        super().__init__()
        self.max_cer = max_cer

    def compare_list(self, record_list: List[str], result_list: List[Tuple[str, float]], biblio_class: str) -> Tuple[int, List[int], List[float]]:
        hits = 0
        labels = []
        conf = []
        ignore_record_i = []
        result_list = sorted(result_list, key=lambda x: x[1], reverse=True)
        for k, (result_item, result_conf) in enumerate(result_list):
            record_found = False
            for record_i, record_item in enumerate(record_list):
                if record_i in ignore_record_i:
                    continue
                if (len(record_item) >= len(result_item)) or (len(result_item) > len(record_item) * 2):
                    cer = self.get_cer(record_item, result_item)
                else:
                    cer = self.get_substr_cer(record_item, result_item)
                    standard_cer = self.get_cer(record_item, result_item)
                    if cer <= self.max_cer and standard_cer > self.max_cer:
                        logger.info(f"sttribute: {biblio_class}, CER: {cer}, standard CER: {standard_cer}, record: {record_item}, result: {result_item}")
                # TP
                if cer <= self.max_cer:
                    hits += 1
                    labels.append(1)
                    conf.append(result_conf)
                    ignore_record_i.append(record_i)
                    record_found = True
                    break
            # FP
            if not record_found:
                labels.append(0)
                conf.append(result_conf)
        # FN
        for record_i, record_item in enumerate(record_list):
            if record_i in ignore_record_i:
                continue
            labels.append(-1)
            conf.append(0.0)

        return hits, labels, conf



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


def compute_ap(true, score):
    """
    Computes Average Precision (AP) using all recall points, explicitly handles False Negatives (FN).

    Args:
        true (np.array): Binary array (1 for TP, 0 for FP, -1 for FN).
        score (np.array): Confidence scores (0 for FN, >0 for detected samples).

    Returns:
        ap (float): Average Precision.
    """

    # Step 1: Separate False Negatives (FN) from detections
    fn_mask = (true == -1)  # FN samples have true = -1
    num_fn = np.sum(fn_mask)  # Count of FN samples

    # Step 2: Filter out FN samples for sorting and precision-recall computation
    valid_mask = ~fn_mask  # Keep only detected samples (TP/FP)
    true_detected = true[valid_mask]
    score_detected = score[valid_mask]

    # Step 3: Sort detected samples by confidence score (descending)
    sorted_indices = np.argsort(-score_detected)  # Sort high → low
    true_sorted = true_detected[sorted_indices]

    # Step 4: Compute cumulative TP and FP
    tp_cumsum = np.cumsum(true_sorted)  # Cumulative TP
    fp_cumsum = np.cumsum(1 - true_sorted)  # Cumulative FP

    # Step 5: Compute Precision and Recall
    total_gt = np.sum(true == 1) + num_fn  # Total GT = TP + FN
    recall = tp_cumsum / total_gt  # Adjusted Recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)  # Precision as TP / (TP + FP)
    pr_sum = (precision + recall)
    pr_sum[pr_sum == 0] = 0.0000001
    f1 = 2 * (precision * recall) / pr_sum

    # Step 6: Apply Monotonic Interpolation (VOC 2012-style smoothing)
    precision_interp = np.maximum.accumulate(precision[::-1])[::-1]

    # Step 7: Compute AP as area under PR curve using trapezoidal integration
    ap = np.sum((recall[1:] - recall[:-1]) * precision_interp[1:]) if len(recall) > 1 else 0.0

    return ap, precision_interp, recall, f1, score_detected[sorted_indices]