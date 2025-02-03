import argparse
import json
import logging
import os

from biblio_dataset.biblio_evaluators import CERBiblioEvaluator, BaseBiblioEvaluator, BiblioResult, \
    normalize_biblio_result
from biblio_dataset.create_biblio_dataset import BiblioRecord, normalize_biblio_record

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate biblio dataset")

    parser.add_argument('--record-dir', type=str, required=True, help="Ground truth directory with biblio records")
    parser.add_argument('--result-dir', type=str, required=True, help="Engine directory with biblio results")

    parser.add_argument('--normalize-record', action='store_true')
    parser.add_argument('--normalize-result', action='store_true')

    parser.add_argument('--evaluator', type=str, default='CER-01', choices=['CER-01'])

    parser.add_argument("--logging-level", default="INFO", choices=["ERROR", "WARNING", "INFO", "DEBUG"])

    return parser.parse_args()


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level)


    biblio_records = load_biblio_records(args.record_dir)
    logger.info(f'{len(biblio_records)} biblio records loaded from {args.record_dir}')
    biblio_results = load_biblio_results(args.result_dir)
    logger.info(f'{len(biblio_results)} biblio results loaded from {args.result_dir}')

    if len(biblio_records) != len(biblio_results):
        logger.warning(f"Number of records and results is different: {len(biblio_records)} vs {len(biblio_results)}")

    if args.normalize_record:
        logger.info("")
        logger.info("Normalizing record records...")
        biblio_records = [BiblioRecord.model_validate(normalize_biblio_record(biblio_record)) for biblio_record in biblio_records]
        logger.info("Normalization done")

    if args.normalize_result:
        logger.info("")
        logger.info("Normalizing result records...")
        biblio_results = [normalize_biblio_result(biblio_result) for biblio_result in biblio_results]
        logger.info("Normalization done")

    evaluator = eval_biblio_records(biblio_records, biblio_results, args.evaluator)
    stats = evaluator.get_stats()

    logger.info("")
    log_stats(stats)


def load_biblio_records(input_dir: str):
    biblio_records = []
    for file in os.listdir(input_dir):
        if not file.endswith('.json'):
            continue
        with open(os.path.join(input_dir, file), 'r') as f:
            biblio_records.append(BiblioRecord.model_validate(json.load(f)))
    return biblio_records

def load_biblio_results(input_dir: str):
    biblio_results = []
    for file in os.listdir(input_dir):
        if not file.endswith('.json'):
            continue
        with open(os.path.join(input_dir, file), 'r') as f:
            biblio_results.append(BiblioResult.model_validate(json.load(f)))
    return biblio_results


def eval_biblio_records(biblio_records, biblio_results, evaluator_name) -> BaseBiblioEvaluator:
    biblio_results = sorted(biblio_results, key=lambda x: x.library_id)
    library_id_to_biblio_record = {record.library_id: record for record in biblio_records}

    logger.info(f"Evaluator: {evaluator_name}")
    if evaluator_name == 'CER-01':
        evaluator = CERBiblioEvaluator(max_cer=0.1)
    else:
        raise NotImplementedError(f"Evaluator {evaluator_name} is not implemented")

    for biblio_result in biblio_results:
        biblio_record = library_id_to_biblio_record.get(biblio_result.library_id)
        if biblio_record is None:
            logger.warning(f"Record with library_id={biblio_result.library_id} not found in ground truth, skipping -> the final results will be incorrect!")
            continue
        evaluator.compare_biblio_record_to_result(biblio_record, biblio_result)

    return evaluator


def log_stats(stats):
    header = f"{'Key':>25}  {'GT':<10}{'TP':<10}{'FP':<10}{'Recall':<10}{'Precision':<10}{'F1':<10}"
    separator = "-" * len(header)

    logger.info(header)
    logger.info(separator)

    for key, val in stats.items():
        record = val['record']
        result_true_positive = val['result_true_positive']
        result_false_positive = val['result_false_positive']
        recall = result_true_positive / record if record != 0 else 0
        result = result_true_positive + result_false_positive
        precision = result_true_positive / result if result != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
        logger.info(f"{key:>25}  {record:<10}{result_true_positive:<10}{result_false_positive:<10}{recall:<10.2f}{precision:<10.2f}{f1:<10.2f}")

    total_record = sum([val['record'] for val in stats.values()])
    total_result_true_positive = sum([val['result_true_positive'] for val in stats.values()])
    total_result_false_positive = sum([val['result_false_positive'] for val in stats.values()])
    total_result = total_result_true_positive + total_result_false_positive
    total_recall = total_result_true_positive / total_record if total_record != 0 else 0
    total_precision = total_result_true_positive / total_result if total_result != 0 else 0
    total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall) if total_precision + total_recall != 0 else 0
    acc = total_result_true_positive / total_record if total_record != 0 else 0

    logger.info(separator)
    logger.info(f"{'All':>25}  {total_record:<10}{total_result_true_positive:<10}{total_result_false_positive:<10}{total_recall:<10.2f}{total_precision:<10.2f}{total_f1:<10.2f}")
    logger.info(f"{'Accuracy':>25}  {acc:<10.2f}")

if __name__ == "__main__":
    main()