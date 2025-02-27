import argparse
import logging
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from biblio_dataset.create_biblio_dataset import biblio_record_classes_to_index, biblio_record_classes_to_human_friendly
from biblio_dataset.evaluate_biblio_dataset import load_biblio_records


logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Show biblio stats")

    parser.add_argument('--record-dir', type=str, required=True, help="Ground truth directory with biblio records")

    parser.add_argument('--out-stats', type=str, help="File to save stats")

    parser.add_argument('--show-plot', action='store_true')
    parser.add_argument('--plot-title', type=str, default='Biblio Stats')

    parser.add_argument("--logging-level", default="INFO", choices=["ERROR", "WARNING", "INFO", "DEBUG"])

    return parser.parse_args()


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level)


    biblio_records = load_biblio_records(args.record_dir)
    logger.info(f'{len(biblio_records)} biblio records loaded from {args.record_dir}')

    stats = defaultdict(int)
    stats_normalized = defaultdict(int)
    for biblio_record in biblio_records:
        biblio_dict = biblio_record.model_dump(exclude_none=True)
        for key, value in biblio_dict.items():
            if 'id' in key:
                continue
            stats_normalized[key] += 1
            if isinstance(value, list):
                stats[key] += len(value)
            else:
                stats[key] += 1

    logger.info("")
    for key, value in stats.items():
        logger.info(f"{key}: {value}")

    if args.out_stats:
        with open(args.out_stats, 'w') as f:
            for key, value in stats.items():
                f.write(f"{key} {value}\n")
        with open(args.out_stats + '_norm', 'w') as f:
            for key, value in stats_normalized.items():
                f.write(f"{key} {value}\n")

    if args.show_plot:
        plot_stats(stats, stats_normalized, args.plot_title)

def plot_stats(stats, stats_normalized, title="Biblio Stats"):
    plt.rcParams.update({"font.size": 10})
    plt.rcParams.update({"figure.figsize": (2.3, 4)})
    plt.rcParams.update({"font.family": "cmr10"})
    plt.rc('axes.formatter', use_mathtext=True)

    labels = []
    for i, (key, val) in enumerate(stats.items()):
        labels += [biblio_record_classes_to_index[key]] * val

    labels_normalized = []
    for i, (key, val) in enumerate(stats_normalized.items()):
        labels_normalized += [biblio_record_classes_to_index[key]] * val

    classes = list(biblio_record_classes_to_human_friendly.values())

    values, bins, bars = plt.hist(labels, bins=range(len(classes) + 1), align="left", rwidth=0.8,
                                  orientation="horizontal", zorder=5, color='white', edgecolor='black', linewidth=1)

    bar_labels = plt.bar_label(bars, fontsize=10, color='black', fmt=' %d')
    for bar_label in bar_labels:
        bar_label.set_font("cmb10")


    plt.yticks(np.arange(len(classes)), [str(i + 1) for i in range(len(classes))], linespacing=1.0)
    plt.ylim(len(classes) - 0.5, -0.6)
    plt.tight_layout()
    plt.subplots_adjust(left=0.12, right=0.95, top=0.97, bottom=0.08)
    plt.xscale("log")
    plt.xticks([1, 10, 100, 1000, 10000], ["1", "10", "100", "1k", "10k"])

    # TEST SET
    plt.xticks([2, 3, 4, 5, 6, 7, 8, 9, 20, 30, 40, 50, 60, 70, 80, 90, 200, 300, 400, 500, 600, 700, 800, 900, 2000, 4000],
               minor=True)
    plt.xlim(1, 8500)

    plt.grid(axis="x", which="both", color="lightgray", linewidth=0.5, zorder=0)
    ax = plt.gca()
    ax.spines['left'].set_zorder(30)
    ax.spines['bottom'].set_zorder(30)

    total_amount = sum(values)

    for category_index, amount in enumerate(values):
        category = classes[category_index].replace('\n', ' ')
        print(f"{category:<30} {int(amount):>5} ({100 * amount / total_amount:.2f} %)")

        amount_norm = stats_normalized[list(biblio_record_classes_to_index.keys())[category_index]]
        # Overlay class names on bars
        plt.text(1.2, category_index + 0.05, category, va='center', ha='left', fontsize=10, color="black", zorder=20)
        #plt.text(amount, category_index, f"{int(amount)}/{int(amount_norm)}", va='center', ha='left', fontsize=10, color="black", zorder=20)

    plt.show()



if __name__ == "__main__":
    main()