import argparse
import logging

import numpy as np
from matplotlib import pyplot as plt



logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Show biblio dates")

    parser.add_argument('--dates', type=str, required=True, help="File with dates, doc_id start_date end_date")

    parser.add_argument("--logging-level", default="INFO", choices=["ERROR", "WARNING", "INFO", "DEBUG"])

    return parser.parse_args()


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level)


    dates = []
    with open(args.dates, 'r') as f:
        for line in f.readlines():
            doc_id, start_date, end_date = line.strip().split()
            if start_date != 'None' and end_date != 'None':
                d = (int(start_date) + int(end_date)) // 2
            elif start_date != 'None':
                d = int(start_date)
            else:
                raise ValueError(f"Invalid date: {line}")
            dates.append(d)

    dates = np.asarray(dates)
    logger.info("Max date: %d", np.max(dates))
    logger.info("Min date: %d", np.min(dates))
    bins = [1485, 1800] + list(range(1820, 2000, 20)) + [2014]
    indexes = np.digitize(dates, bins=bins)

    start_end_count = {}
    for i in range(1, len(bins)):
        start_end_count[i - 1] = (bins[i - 1], bins[i], np.sum(indexes == i))

    years = {}
    intervals = {}
    for i, (start, end, count) in start_end_count.items():
        years[i + 1] = int(count)
        intervals[i] = (int(start), int(end))

    logger.info(f"Years: {years}")
    logger.info(f"Intervals: {intervals}")

    total = sum(years.values())
    years = {k: 100 * v / total for k, v in years.items()}

    logger.info(f"Years: {years}")

    plt.rcParams.update({"font.size": 10})
    plt.rcParams.update({"figure.figsize": (5, 2)})
    # plt.rcParams.update({"figure.figsize": (10, 8)})
    plt.rcParams.update({"font.family": "cmr10"})
    plt.rc('axes.formatter', use_mathtext=True)

    values, bins, bars = plt.hist(list(range(len(years.keys()))), weights=list(years.values()),
                                  bins=list(range(len(years.keys()) + 1)), rwidth=0.8, zorder=5, color='skyblue', edgecolor='black', linewidth=1)
    plt.xticks(list(range(len(years.keys()) + 1)),
               [f'{intervals[i][0]}' for i in range(len(intervals.keys()))] + [str(list(intervals.values())[-1][-1])],
               rotation=45)
    yticks = plt.yticks()
    plt.yticks(yticks[0], [f'{int(y)} %' if int(y) != 0 else '' for y in yticks[0]])
    plt.tight_layout()

    plt.grid(axis="y", which="both", color="lightgray", linewidth=0.5, zorder=0)

    plt.axvline(x=1, color='k', linestyle='--', linewidth=1, alpha=1, zorder=1)
    plt.text(0.8, 20, 'Historic', rotation=90, verticalalignment='bottom', horizontalalignment='center', fontsize=10)
    plt.text(1.3, 20, 'Modern', rotation=90, verticalalignment='bottom', horizontalalignment='center', fontsize=10)


    plt.show()



if __name__ == "__main__":
    main()