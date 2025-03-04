import argparse
import logging

import numpy as np
from matplotlib import pyplot as plt



logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Show biblio libraries")

    parser.add_argument('--libraries', type=str, required=True, help="File with libraries, doc_id library")

    parser.add_argument("--logging-level", default="INFO", choices=["ERROR", "WARNING", "INFO", "DEBUG"])

    return parser.parse_args()


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level)


    libraries = []
    with open(args.libraries, 'r') as f:
        for line in f.readlines():
            doc_id, library = line.strip().split()
            libraries.append(library)



    lib_to_count = {}
    for lib in libraries:
        if lib not in lib_to_count:
            lib_to_count[lib] = 0
        lib_to_count[lib] += 1

    new_lib_count = {}
    other = 0
    for lib, count in lib_to_count.items():
        if count > 20:
            new_lib_count[lib] = count
        else:
            other += count
    lib_to_count = new_lib_count

    lib_to_count = sorted(lib_to_count.items(), key=lambda x: x[1], reverse=True)
    lib_to_count.append(('Other', other))

    total = sum([count for lib, count in lib_to_count])
    weights = [100 * (count / total) for lib, count in lib_to_count]
    classes = [lib.upper() for lib, count in lib_to_count]

    logger.info(f"Libraries: {lib_to_count}")
    logger.info(f"Classes: {classes}")
    logger.info(f"Weights: {weights}")





    plt.rcParams.update({"font.size": 10})
    plt.rcParams.update({"figure.figsize": (5, 2)})
    # plt.rcParams.update({"figure.figsize": (10, 8)})
    plt.rcParams.update({"font.family": "cmr10"})
    plt.rc('axes.formatter', use_mathtext=True)



    values, bins, bars = plt.hist(range(len(classes)), bins=range(len(classes) + 1), weights=weights,
                                  color='skyblue', edgecolor='black', linewidth=1, zorder=5, rwidth=0.8)

    plt.xticks(np.asarray(list(range(len(classes)))) + 0.5, classes, rotation=45)
    yticks = plt.yticks()
    plt.yticks(yticks[0], [f'{int(y)} %' if int(y) != 0 else '' for y in yticks[0]])
    plt.tight_layout()

    plt.grid(axis="y", which="both", color="lightgray", linewidth=0.5, zorder=0)

    plt.show()



if __name__ == "__main__":
    main()