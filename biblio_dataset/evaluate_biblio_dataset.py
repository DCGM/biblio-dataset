import argparse
import json
import logging
import os

from biblio_dataset.create_biblio_dataset import BiblioRecord
from biblio_dataset.create_biblio_dataset import normalize_biblio_record

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Crete biblio dataset")

    parser.add_argument('--gt-dir', type=str, required=True)
    parser.add_argument('--result-dir', type=str, required=True)

    parser.add_argument('--normalize-result', action='store_true')

    parser.add_argument("--logging-level", default="INFO", choices=["ERROR", "WARNING", "INFO", "DEBUG"])

    return parser.parse_args()


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level)


    gt_biblio_records = load_biblio_records(args.gt_dir)
    result_biblio_records = load_biblio_records(args.result_dir)

    if args.normalize_result:
        result_biblio_records = [normalize_biblio_record(record) for record in result_biblio_records]


def eval_biblio_records(gt_biblio_records, result_biblio_records):
    result_biblio_records = sorted(result_biblio_records, key=lambda x: x.task_id)
    task_id_to_gt_biblio_record = {record.task_id: record for record in gt_biblio_records}

    for result_record in result_biblio_records:
        gt_record = task_id_to_gt_biblio_record.get(result_record.task_id)
        if gt_record is None:
            logger.warning(f"Record with task_id={result_record.task_id} not found in ground truth, skipping")
            continue
        compare_biblio_records(gt_record, result_record)


def compare_biblio_records(gt_record, result_record):
    pass


def load_biblio_records(input_dir: str):
    biblio_records = []
    for file in os.listdir(input_dir):
        if not file.endswith('.json'):
            continue
        with open(os.path.join(input_dir, file), 'r') as f:
            biblio_records.append(BiblioRecord.model_validate(json.load(f)))
    return biblio_records