import argparse
import json
import logging
import os
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

from biblio_dataset.biblio_normalizer import BiblioNormalizer
from biblio_dataset.biblio_evaluators import CERBiblioEvaluator, BaseBiblioEvaluator, BiblioResult, compute_ap
from biblio_dataset.create_biblio_dataset import BiblioRecord, biblio_record_classes_to_human_friendly, \
    biblio_record_classes_to_colors
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate biblio dataset")

    parser.add_argument('--record-dir', type=str, required=True, help="Ground truth directory with biblio records")
    parser.add_argument('--result-dir', type=str, required=True, help="Engine directory with biblio results")

    parser.add_argument('--normalize-record', action='store_true')
    parser.add_argument('--normalize-result', action='store_true')
    parser.add_argument('--lowercase-record', action='store_true')
    parser.add_argument('--lowercase-result', action='store_true')
    parser.add_argument('--remove-diacritics-record', action='store_true')
    parser.add_argument('--remove-diacritics-result', action='store_true')

    parser.add_argument('--confidence-threshold', type=float, default=0.25, help="For recall and precision")

    parser.add_argument('--evaluator', type=str, default='CER-01', choices=['CER-01'])

    parser.add_argument('--show-prc-plot', action='store_true')
    parser.add_argument('--prc-plot-title', type=str, default='Precision-Recall Curve')
    parser.add_argument('--show-f1-plot', action='store_true')
    parser.add_argument('--f1-plot-title', type=str, default='F1 Curve')

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
        record_normalizer = BiblioNormalizer(lowercase=args.lowercase_record, remove_diacritics=args.remove_diacritics_record)
        logger.info("")
        logger.info("Normalizing record records...")
        biblio_records = [BiblioRecord.model_validate(record_normalizer.normalize_biblio_record(biblio_record)) for biblio_record in biblio_records]
        logger.info("Normalization done")

    if args.normalize_result:
        result_normalizer = BiblioNormalizer(lowercase=args.lowercase_result, remove_diacritics=args.remove_diacritics_result)
        logger.info("")
        logger.info("Normalizing result records...")
        biblio_results = [result_normalizer.normalize_biblio_result(biblio_result) for biblio_result in biblio_results]
        logger.info("Normalization done")

    evaluator = eval_biblio_records(biblio_records, biblio_results, args.evaluator)
    stats = evaluator.get_stats()

    logger.info("")
    log_stats(stats, confidence_threshold=args.confidence_threshold,
              show_prc_plot=args.show_prc_plot, prc_plot_title=args.prc_plot_title,
              show_f1_plot=args.show_f1_plot, f1_plot_title=args.f1_plot_title)


def load_biblio_records(input_dir: str):
    biblio_records = []
    for file in os.listdir(input_dir):
        if not file.endswith('.json'):
            continue
        file_path = os.path.join(input_dir, file)
        with open(file_path, 'r') as f:
            logger.debug(f"Loading biblio record: {file_path}")
            biblio_records.append(BiblioRecord.model_validate(json.load(f)))
    return biblio_records

def load_biblio_results(input_dir: str):
    biblio_results = []
    for file in os.listdir(input_dir):
        if not file.endswith('.json'):
            continue
        file_path = os.path.join(input_dir, file)
        with open(file_path, 'r') as f:
            logger.debug(f"Loading biblio result: {file_path}")
            biblio_results.append(BiblioResult.model_validate(json.load(f)))
    return biblio_results


def eval_biblio_records(biblio_records, biblio_results, evaluator_name) -> BaseBiblioEvaluator:
    biblio_results = sorted(biblio_results, key=lambda x: x.library_id)
    library_id_to_biblio_result = {result.library_id: result for result in biblio_results}

    logger.info(f"Evaluator: {evaluator_name}")
    if evaluator_name == 'CER-01':
        evaluator = CERBiblioEvaluator(max_cer=0.1)
    else:
        raise NotImplementedError(f"Evaluator {evaluator_name} is not implemented")

    for biblio_record in biblio_records:
        biblio_result = library_id_to_biblio_result.get(biblio_record.library_id)
        if biblio_result is None:
            logger.warning(f"Result not found for record: {biblio_record.library_id}, replacing with empty result")
            biblio_result = BiblioResult(library_id=biblio_record.library_id)
        evaluator.compare_biblio_record_to_result(biblio_record, biblio_result)

    return evaluator


def log_stats(stats, confidence_threshold=0.25, show_prc_plot=False, prc_plot_title='Precision-Recall Curve',
              show_f1_plot=False, f1_plot_title='F1 Curve'):
    header = f"{'Key':>25}  {'GT':<10}{'TP':<10}{'FP':<10}{'FN':<10}{'Recall':<10}{'Precision':<10}{'F1':<10}{'AP':<10}"
    separator = "-" * len(header)

    logger.info(header)
    logger.info(separator)

    avg_recall = []
    avg_precision = []
    avg_f1 = []
    avg_true_aps = []

    precision_all = []
    recall_all = []
    f1_all = []
    confidences_all = []

    true_positive_all = 0

    for key, val in stats.items():
        record = val['record']
        result_true_positive = 0
        result_false_positive = 0
        result_false_negative = 0
        for label, conf in zip(val['result_label'], val['result_conf']):
            if label == -1 or label == 1 and conf < confidence_threshold:
                result_false_negative += 1
                continue
            if conf < confidence_threshold:
                continue
            if label == 1:
                result_true_positive += 1
            else:
                result_false_positive += 1
        true_positive_all += result_true_positive

        recall_true = deepcopy(val['result_label'])
        recall_pred = deepcopy(val['result_conf'])
        for i in range(len(recall_true)):
            if recall_true[i] == -1:
                recall_true[i] = 1
                recall_pred[i] = 0
            else:
                if recall_pred[i] < confidence_threshold:
                    recall_pred[i] = 0
                else:
                    recall_pred[i] = 1

        precision_true = []
        precision_pred = []
        for label, conf in zip(val['result_label'], val['result_conf']):
            if label != -1:
                precision_true.append(label)
                if conf < confidence_threshold:
                    precision_pred.append(0)
                else:
                    precision_pred.append(1)

        recall = recall_score(recall_true, recall_pred)
        precision = precision_score(precision_true, precision_pred)
        f1 = f1_score(recall_true, recall_pred)

        #check recall, precision, f1
        check_recall = result_true_positive / (result_true_positive + result_false_negative) if result_true_positive + result_false_negative != 0 else 0
        if recall - check_recall > 0.01:
            logger.warning(f"Recall mismatch: {recall} vs {check_recall}")
        else:
            logger.debug(f"Recall {recall} vs {check_recall}")
        check_precision = result_true_positive / (result_true_positive + result_false_positive) if result_true_positive + result_false_positive != 0 else 0
        if precision - check_precision > 0.01:
            logger.warning(f"Precision mismatch: {precision} vs {check_precision}")
        else:
            logger.debug(f"Precision {precision} vs {check_precision}")
        check_f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
        if f1 - check_f1 > 0.01:
            logger.warning(f"F1 mismatch: {f1} vs {check_f1}")
        else:
            logger.debug(f"F1 mismatch {f1} vs {check_f1}")


        #aps = average_precision_score(np.asarray(recall_true).reshape(-1, 1),
        #                              np.asarray(val['result_conf']).reshape(-1, 1))
        true_aps, prc_precision, prc_recall, prc_f1, prc_confidences = compute_ap(np.asarray(val['result_label']), np.asarray(val['result_conf']))
        precision_all.append(prc_precision)
        recall_all.append(prc_recall)
        f1_all.append(prc_f1)
        confidences_all.append(prc_confidences)

        avg_recall.append(recall)
        avg_precision.append(precision)
        avg_f1.append(f1)
        #avg_aps.append(aps)
        avg_true_aps.append(true_aps)

        logger.info(f"{key:>25}  {record:<10}{result_true_positive:<10}{result_false_positive:<10}{result_false_negative:<10}{recall:<10.2f}{precision:<10.2f}{f1:<10.2f}{true_aps:<10.2f}")

    total_record = sum([val['record'] for val in stats.values()])
    #total_result_false_positive = sum([val['result_false_positive'] for val in stats.values()])
    #total_result = total_result_true_positive + total_result_false_positive
    #total_recall = total_result_true_positive / total_record if total_record != 0 else 0
    #total_precision = total_result_true_positive / total_result if total_result != 0 else 0
    #total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall) if total_precision + total_recall != 0 else 0
    acc = true_positive_all / total_record if total_record != 0 else 0

    logger.info(separator)
    logger.info(f"{'AVG':>25}  {"N/A":<10}{"N/A":<10}{"N/A":<10}{"N/A":<10}{np.mean(avg_recall):<10.2f}{np.mean(avg_precision):<10.2f}{np.mean(avg_f1):<10.2f}{np.mean(avg_true_aps):<10.2f}")
    logger.info(f"{'Accuracy':>25}  {acc:<10.2f}")

    avg_confidences_c, avg_f1_c = compute_average_curve(confidences_all, f1_all)
    best_confidence = avg_confidences_c[np.argmax(avg_f1_c)]
    logger.info(f"Best confidence threshold  {best_confidence:<10.2f}")

    logger.info('Recall:')
    logger.info(f'{int(np.mean(avg_recall) * 100)} & ' + latex_string(avg_recall))
    logger.info('Precision:')
    logger.info(f'{int(np.mean(avg_precision) * 100)} & ' + latex_string(avg_precision))
    logger.info('F1:')
    logger.info(f'{int(np.mean(avg_f1) * 100)} & ' + latex_string(avg_f1))
    logger.info('AP:')
    logger.info(f'{int(np.mean(avg_true_aps) * 100)} & ' + latex_string(avg_true_aps))
    logger.info('Accuracy:')
    logger.info(f'{int(acc * 100)}')

    if show_prc_plot:
        plot_prc_f1(precision_all, recall_all, stats.keys(), title=prc_plot_title)

    if show_f1_plot:
        plot_prc_f1(f1_all, confidences_all, stats.keys(), title=f1_plot_title, f1=True)


def latex_string(vals):
    vals = np.asarray(vals) * 100
    return ' & '.join([f'{int(x)}' for x in vals])


def plot_prc_f1(y, x, attributes, title='Precision-Recall Curve', f1=False):
    # Create figure and plot the Precision-Recall curves

    plt.rcParams.update({"font.size": 10})
    plt.rcParams.update({"figure.figsize": (5, 4)})
    plt.rcParams.update({"font.family": "cmr10"})
    plt.rc('axes.formatter', use_mathtext=True)

    # Store legend handles separately
    legend_handles = []

    # Loop through the precision, recall, and attribute names
    for i, (precision, recall, attr) in enumerate(zip(y, x, attributes)):
        #recall = np.append(recall, recall[-1])  # Add an extra value to the end of the recall array
        #precision = np.append(precision, 0)  # Add an extra value to the end of the precision array
        # Use the attribute's mapped human-readable name
        label = biblio_record_classes_to_human_friendly.get(attr, attr)  # Default to the attribute itself if no mapping exists
        color = biblio_record_classes_to_colors.get(attr, 'black')  # Default to black if no mapping exists
        # Plot with the corresponding color from the colormap
        plt.plot(recall, precision, linestyle='-', label=label, color=color, linewidth=1.7)
        legend_handles.append(Line2D([], [], marker='s', linestyle='None', color=color, label=label, markersize=4))

    if f1:
        avg_x, avg_y = compute_average_curve(x, y)
        plt.plot(avg_x, avg_y, linestyle='--', label='Average', color='black', linewidth=1.7)
        legend_handles.append(Line2D([], [], marker='s', linestyle='None', color='black', label='Average', markersize=4))

    # Labels, title, and legend
    if not f1:
        plt.xlabel("Recall")
        plt.ylabel("Precision")
    else:
        plt.xlabel("Confidence")
        plt.ylabel("F1")
    plt.title(title, x=0.72)

    # Set x and y ticks at intervals of 0.1
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))

    # Position the legend below the plot
    plt.legend(handles=legend_handles,
               loc='upper left',
               bbox_to_anchor=(1, 1.02),
               ncol=1,
               handlelength=0,
               markerscale=1.5,
               borderpad=0.9,
               prop={'size': 10})

    if not f1:
        # Set x-axis limit to go up to 1 with some padding
        plt.xlim(-0.05, 1.05)  # 1.05 provides a little padding beyond 1
        plt.ylim(0.2, 1.05)  # 1.05 provides a little padding beyond 1
    else:
        # Set x-axis limit to go up to 1 with some padding
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)

    plt.grid(True, which="both", linestyle="-", alpha=0.5)

    # Adjust layout for tight display
    plt.tight_layout()

    # Show the plot
    plt.show()

    return


def compute_average_curve(x_list, y_list):
    """
    Computes the average curve where any y-value outside the range of others is set to zero.

    Parameters:
        x_list (list of numpy arrays): A list of x values for each curve.
        y_list (list of numpy arrays): A list of y values for each curve.

    Returns:
        avg_y (numpy array): The averaged y values, with invalid points set to zero.
    """
    # Find the union of all x values from all curves
    all_x_vals = np.unique(np.concatenate(x_list))

    # Interpolate each curve to the common x_vals using interp1d
    interpolated_ys = []

    x_list = [np.asarray(x) for x in x_list]
    y_list = [np.asarray(y) for y in y_list]

    for x_vals, y_vals in zip(x_list, y_list):
        if x_vals.size == 0:
            continue
        if y_vals.size == 0:
            continue
        # Create an interpolator for each curve (using linear interpolation)
        interpolator = interp1d(x_vals, y_vals, kind='linear', bounds_error=False, fill_value="extrapolate")

        # Interpolate the curve over the common x_vals
        interpolated_ys.append(interpolator(all_x_vals))

    # Convert list of interpolated y-values into a numpy array
    interpolated_ys = np.array(interpolated_ys)

    # Compute the average for each x, but set to zero if any curve is outside the range
    avg_y = np.zeros_like(all_x_vals)

    # Iterate through each x position
    for i, x in enumerate(all_x_vals):
        # Get the min and max y-values for the current x (across all curves)
        min_y = np.min(interpolated_ys[:, i])
        max_y = np.max(interpolated_ys[:, i])

        # If any y-value is outside the range [min_y, max_y], set it to zero
        valid_values = np.all((interpolated_ys[:, i] >= min_y) & (interpolated_ys[:, i] <= max_y))

        if valid_values:
            # Compute the average if the values are valid
            avg_y[i] = np.mean(interpolated_ys[:, i])
        else:
            # Set it to zero if any curve is out of range
            avg_y[i] = 0

    return all_x_vals, avg_y




if __name__ == "__main__":
    main()