import argparse
import json
import logging
import os

from biblio_dataset.biblio_evaluators import convert_alto_match_to_biblio_results
from biblio_dataset.create_biblio_dataset import check_biblio_record
from detector_wrapper.parsers.detector_parser import DetectorParser
from detector_wrapper.parsers.pero_ocr import ALTOMatch


logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLO + PERO-OCR engine")

    parser.add_argument('--yolo-export-dir', type=str, required=True)
    parser.add_argument('--yolo-yaml', type=str)
    parser.add_argument('--alto-export-dir', type=str, required=True)

    parser.add_argument('--min-alto-word-area-in-detection-to-match', type=float, default=0.65)

    parser.add_argument('--output-dir', type=str)

    parser.add_argument("--logging-level", default="INFO", choices=["ERROR", "WARNING", "INFO", "DEBUG"])

    return parser.parse_args()


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level)

    detector_parser = DetectorParser()
    detector_parser.parse_yolo(args.yolo_export_dir, args.yolo_yaml)

    alto_match = ALTOMatch(detector_parser,
                           args.alto_export_dir,
                           min_alto_word_area_in_detection_to_match=args.min_alto_word_area_in_detection_to_match)
    alto_match.match()

    biblio_results = convert_alto_match_to_biblio_results(alto_match)


    [check_biblio_record(biblio_record) for biblio_record in biblio_results]

    if args.output_dir is not None:
        logger.info(f"Saving {len(biblio_results)} biblio records to {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        for biblio_record in biblio_results:
            with open(os.path.join(args.output_dir, f'{biblio_record.library_id}.json'), 'w') as f:
                json.dump(biblio_record.model_dump(exclude_none=True), f, indent=4, default=str, ensure_ascii=False)




if __name__ == "__main__":
    main()
