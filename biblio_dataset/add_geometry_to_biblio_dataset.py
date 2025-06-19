import argparse
import json
import logging
import os
import sys
import Levenshtein
from collections import defaultdict

from biblio_dataset.create_biblio_dataset import BiblioRecordWithGeometry, \
    label_studio_classes_to_biblio_record_classes
from biblio_dataset.evaluate_biblio_dataset import load_biblio_records
from detector_wrapper.parsers.detector_parser import DetectorParser
from detector_wrapper.parsers.pero_ocr import ALTOMatch

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Crete biblio dataset")

    parser.add_argument('--record-dir', type=str, required=True, help="Ground truth directory with biblio records")
    parser.add_argument('--alto-export-dir', type=str, required=True)
    parser.add_argument('--yolo-export-dir', type=str, required=True)
    parser.add_argument('--yolo-yaml', type=str, required=True)
    parser.add_argument('--image-dir', type=str, required=True, help="Directory with images")


    parser.add_argument('--min-alto-word-area-in-detection-to-match', type=float, default=0.5)

    parser.add_argument('--output-dir', type=str)

    parser.add_argument("--logging-level", default="INFO", choices=["ERROR", "WARNING", "INFO", "DEBUG"])

    return parser.parse_args()


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level)

    logger.info(' '.join(sys.argv))
    logger.info('')

    detector_parser = DetectorParser()
    detector_parser.parse_yolo(args.yolo_export_dir, args.yolo_yaml, args.image_dir, default_confidence=1)

    yolo_page_mapping = {page.id: page for page in detector_parser.annotated_pages}

    alto_match = ALTOMatch(detector_parser,
                           args.alto_export_dir,
                           min_alto_word_area_in_detection_to_match=args.min_alto_word_area_in_detection_to_match)
    alto_match.match()
    matched_page_mapping = {matched_page.detector_parser_page.id: matched_page for matched_page in alto_match.matched_pages}

    biblio_records = load_biblio_records(args.record_dir)
    logger.info(f'{len(biblio_records)} biblio records loaded from {args.record_dir}')

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    for biblio_record in biblio_records:
        biblio_record = biblio_record.model_dump(exclude_none=True)
        biblio_record_with_geometry = BiblioRecordWithGeometry(task_id=biblio_record['task_id'],
                                                               library_id=biblio_record['library_id'])
        biblio_record_with_geometry = biblio_record_with_geometry.model_dump(exclude_none=False)
        needs_manual_resolution = False
        if biblio_record['library_id'] not in matched_page_mapping:
            logger.warning(f'Biblio record {biblio_record['library_id']} not found in matched pages.')
            needs_manual_resolution = True
        else:
            matched_page = matched_page_mapping[biblio_record['library_id']]
            detection_mapping = defaultdict(list)
            for matched_detection in matched_page.matched_detections:
                detection_mapping[label_studio_classes_to_biblio_record_classes[matched_detection.get_class()]].append(matched_detection)

            for class_name, val in biblio_record.items():
                if class_name == 'task_id' or class_name == 'library_id':
                    continue
                if len(val) == 0:
                    logger.warning(f'Biblio record {biblio_record.library_id} has empty value for class {class_name}. Skipping.')
                    continue
                if class_name in detection_mapping:
                    # 1 to 1 mapping
                    if (isinstance(val, str) and len(detection_mapping[class_name]) == 1) or \
                        (isinstance(val, list) and len(val) == 1 and class_name in detection_mapping and len(detection_mapping[class_name]) == 1):
                        det = (int(detection_mapping[class_name][0].detector_parser_annotated_bounding_box.x),
                               int(detection_mapping[class_name][0].detector_parser_annotated_bounding_box.y),
                               int(detection_mapping[class_name][0].detector_parser_annotated_bounding_box.width),
                               int(detection_mapping[class_name][0].detector_parser_annotated_bounding_box.height))
                        if isinstance(val, list):
                            biblio_record_with_geometry[class_name] = [[val[0], det]]
                        else:
                            biblio_record_with_geometry[class_name] = [val, det]
                    # many to many mapping
                    # elements of val contains text
                    # elements of detection_mapping[class_name] contains text that can be accessed via get_text()
                    # compare all gt_text with all detection.get_text() and find the best match based on levenshtein distance
                    # matches are exclusive, i.e. each gt_text and detection can be used only once
                    elif isinstance(val, list) and len(val) == len(detection_mapping[class_name]):
                        used_detections = set()
                        for gt_text in val:
                            best_match = None
                            best_distance = float('inf')
                            for detection in detection_mapping[class_name]:
                                if detection in used_detections:
                                    continue
                                distance = Levenshtein.distance(gt_text, detection.get_text())
                                if distance < best_distance:
                                    best_distance = distance
                                    best_match = detection
                            if best_match is not None:
                                if biblio_record_with_geometry[class_name] is None:
                                    biblio_record_with_geometry[class_name] = []
                                biblio_record_with_geometry[class_name].append([gt_text, (int(best_match.detector_parser_annotated_bounding_box.x),
                                                                                          int(best_match.detector_parser_annotated_bounding_box.y),
                                                                                          int(best_match.detector_parser_annotated_bounding_box.width),
                                                                                          int(best_match.detector_parser_annotated_bounding_box.height))])
                                used_detections.add(best_match)
                else:
                    logger.warning(f'Biblio record {biblio_record['library_id']} has class {class_name} but no detections found. RESOLVE MANUALLY')
                    needs_manual_resolution = True

        if needs_manual_resolution:
            logger.warning(f'Biblio record {biblio_record['library_id']} needs manual resolution.')
            logger.info(yolo_page_mapping[biblio_record['library_id']].bounding_boxes)

        if args.output_dir is not None:
            with open(os.path.join(args.output_dir, f'{biblio_record['library_id']}.json'), 'w') as f:
                json.dump(BiblioRecordWithGeometry.model_validate(biblio_record_with_geometry).model_dump(exclude_none=True), f, indent=4, default=str, ensure_ascii=False)


if __name__ == "__main__":
    main()
