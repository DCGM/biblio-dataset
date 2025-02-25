import argparse
import json
import logging
import os
import sys

from detector_wrapper.parsers.detector_parser import DetectorParser
from detector_wrapper.parsers.pero_ocr import ALTOMatch

from typing import List, Optional
from pydantic import BaseModel

class BiblioRecord(BaseModel):
    task_id: str
    library_id: str

    title: Optional[str] = None
    subTitle: Optional[str] = None
    partName: Optional[str] = None
    partNumber: Optional[str] = None
    seriesName: Optional[str] = None
    seriesNumber: Optional[str] = None

    edition: Optional[str] = None
    placeTerm: Optional[str] = None
    dateIssued: Optional[str] = None

    publisher: Optional[List[str]] = None
    manufacturePublisher: Optional[str] = None
    manufacturePlaceTerm: Optional[str] = None

    author: Optional[List[str]] = None
    illustrator: Optional[List[str]] = None
    translator: Optional[List[str]] = None
    editor: Optional[List[str]] = None


label_studio_classes_to_biblio_record_classes = {
    'titulek': 'title',
    'podtitulek': 'subTitle',
    'nazev dilu': 'partName',
    'dil': 'partNumber',
    'serie': 'seriesName',
    'cislo serie': 'seriesNumber',

    'vydani': 'edition',
    'misto vydani': 'placeTerm',
    'datum vydani': 'dateIssued',

    # nakladatel and vydavatel are mapped to the same class publisher
    'nakladatel': 'publisher',
    'vydavatel': 'publisher',
    'tiskar': 'manufacturePublisher',
    'misto tisku': 'manufacturePlaceTerm',

    'autor': 'author',
    'ilustrator': 'illustrator',
    'prekladatel': 'translator',
    'editor': 'editor'
}


logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Crete biblio dataset")

    parser.add_argument('--label-studio-export-json', type=str, required=True)
    parser.add_argument('--alto-export-dir', type=str, required=True)

    parser.add_argument('--min-alto-word-area-in-detection-to-match', type=float, default=0.5)

    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--lowercase', action='store_true')
    parser.add_argument('--remove-diacritics', action='store_true')

    parser.add_argument('--output-dir', type=str)

    parser.add_argument('--eval-ids', type=str, help='File, where each line is id')

    parser.add_argument("--logging-level", default="INFO", choices=["ERROR", "WARNING", "INFO", "DEBUG"])

    return parser.parse_args()


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level)
    if args.output_dir is not None:
        if args.normalize:
            file_handler = logging.FileHandler(os.path.join('/'.join(args.output_dir.split('/')[:-1]), 'create_biblio_dataset_normalize.log'))
        else:
            file_handler = logging.FileHandler(os.path.join('/'.join(args.output_dir.split('/')[:-1]), 'create_biblio_dataset.log'))
        file_handler.setLevel(args.logging_level)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

    logger.info(' '.join(sys.argv))
    logger.info('')

    detector_parser = DetectorParser()
    detector_parser.parse_label_studio(label_studio_export=args.label_studio_export_json,
                                       class_remapping={'vydavatel': 'nakladatel'},
                                       run_checks=True)

    alto_match = ALTOMatch(detector_parser,
                           args.alto_export_dir,
                           min_alto_word_area_in_detection_to_match=args.min_alto_word_area_in_detection_to_match)
    alto_match.match()

    biblio_records = convert_alto_match_to_biblio_records(alto_match)

    if args.normalize:
        # this import must be here due to circular import error
        from biblio_normalizer import BiblioNormalizer
        biblio_record_normalizer = BiblioNormalizer(lowercase=args.lowercase, remove_diacritics=args.remove_diacritics)
        biblio_records = [biblio_record_normalizer.normalize_biblio_record(biblio_record) for biblio_record in biblio_records]

    [check_biblio_record(biblio_record) for biblio_record in biblio_records]

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.eval_ids is None:
            for biblio_record in biblio_records:
                with open(os.path.join(args.output_dir, f'{biblio_record.library_id}.json'), 'w') as f:
                    json.dump(biblio_record.model_dump(exclude_none=True), f, indent=4, default=str, ensure_ascii=False)
        else:
            os.makedirs(os.path.join(args.output_dir, 'eval'), exist_ok=True)
            os.makedirs(os.path.join(args.output_dir, 'train'), exist_ok=True)
            with open(args.eval_ids, 'r') as f:
                eval_ids = [line.strip() for line in f.readlines()]
            for biblio_record in biblio_records:
                if biblio_record.library_id in eval_ids:
                    with open(os.path.join(args.output_dir, 'eval', f'{biblio_record.library_id}.json'), 'w') as f:
                        json.dump(biblio_record.model_dump(exclude_none=True), f, indent=4, default=str, ensure_ascii=False)
                else:
                    with open(os.path.join(args.output_dir, 'train', f'{biblio_record.library_id}.json'), 'w') as f:
                        json.dump(biblio_record.model_dump(exclude_none=True), f, indent=4, default=str, ensure_ascii=False)



def check_biblio_record(biblio_record: BiblioRecord):
    #check for duplicat values in list fields
    if biblio_record.publisher is not None:
        if len(biblio_record.publisher) != len(set(biblio_record.publisher)):
            logger.warning(f"Duplicate values in publisher field: {biblio_record.publisher}, id: {biblio_record.task_id}")
    if biblio_record.author is not None:
        if len(biblio_record.author) != len(set(biblio_record.author)):
            logger.warning(f"Duplicate values in author field: {biblio_record.author}, id: {biblio_record.task_id}")
    if biblio_record.illustrator is not None:
        if len(biblio_record.illustrator) != len(set(biblio_record.illustrator)):
            logger.warning(f"Duplicate values in illustrator field: {biblio_record.illustrator}, id: {biblio_record.task_id}")
    if biblio_record.translator is not None:
        if len(biblio_record.translator) != len(set(biblio_record.translator)):
            logger.warning(f"Duplicate values in translator field: {biblio_record.translator}, id: {biblio_record.task_id}")
    if biblio_record.editor is not None:
        if len(biblio_record.editor) != len(set(biblio_record.editor)):
            logger.warning(f"Duplicate values in editor field: {biblio_record.editor}, id: {biblio_record.task_id}")





def convert_alto_match_to_biblio_records(alto_match: ALTOMatch):
    biblio_records = []
    for matched_page in alto_match.matched_pages:
        biblio_record = {'task_id': str(matched_page.detector_parser_page.id),
                         'library_id': os.path.splitext(matched_page.detector_parser_page.image_filename)[0]}
        for matched_detection in matched_page.matched_detections:
            detection_class = matched_detection.get_class()
            if detection_class not in label_studio_classes_to_biblio_record_classes:
                logger.warning(f"Skipping detection with class {detection_class}, label studio task id {matched_page.detector_parser_page.id}")
                continue
            biblio_record_class = label_studio_classes_to_biblio_record_classes[detection_class]
            if biblio_record_class in ['publisher', 'author', 'illustrator', 'translator', 'editor']:
                if biblio_record.get(biblio_record_class) is None:
                    biblio_record[biblio_record_class] = []
                biblio_record[biblio_record_class].append(matched_detection.get_text())
            else:
                if biblio_record_class not in biblio_record:
                    biblio_record[biblio_record_class] = matched_detection.get_text()
                else:
                    logger.warning(f"Multiple values for {biblio_record_class} in label studio task id {matched_page.detector_parser_page.id}")
        biblio_record = BiblioRecord.model_validate(biblio_record)
        biblio_records.append(biblio_record)
    return biblio_records


if __name__ == "__main__":
    main()
