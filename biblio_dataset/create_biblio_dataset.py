import argparse
import json
import logging
import os
import re

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

    parser.add_argument('--min-alto-word-area-in-detection-to-match', type=float, default=0.65)

    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--output-dir', type=str)

    parser.add_argument("--logging-level", default="INFO", choices=["ERROR", "WARNING", "INFO", "DEBUG"])

    return parser.parse_args()


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level)

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
        biblio_records = [normalize_biblio_record(biblio_record) for biblio_record in biblio_records]

    [check_biblio_record(biblio_record) for biblio_record in biblio_records]

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        for biblio_record in biblio_records:
            with open(os.path.join(args.output_dir, f'{biblio_record.library_id}.json'), 'w') as f:
                json.dump(biblio_record.model_dump(exclude_none=True), f, indent=4, default=str)



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


def normalize_biblio_record(biblio_record: BiblioRecord):
    normalized_biblio_record = BiblioRecord(task_id=biblio_record.task_id,
                                            library_id=biblio_record.library_id)

    if biblio_record.title is not None:
        normalized_biblio_record.title = normalize_title(biblio_record.title)

    if biblio_record.subTitle is not None:
        normalized_biblio_record.subTitle = normalize_sub_title(biblio_record.subTitle)

    if biblio_record.partName is not None:
        normalized_biblio_record.partName = normalize_part_name(biblio_record.partName)

    if biblio_record.partNumber is not None:
        normalized_biblio_record.partNumber = normalize_part_number(biblio_record.partNumber)

    if biblio_record.seriesName is not None:
        normalized_biblio_record.seriesName = normalize_series_name(biblio_record.seriesName)

    if biblio_record.seriesNumber is not None:
        normalized_biblio_record.seriesNumber = normalize_series_number(biblio_record.seriesNumber)

    if biblio_record.edition is not None:
        normalized_biblio_record.edition = normalize_edition(biblio_record.edition)

    if biblio_record.placeTerm is not None:
        normalized_biblio_record.placeTerm = normalize_place_term(biblio_record.placeTerm)

    if biblio_record.dateIssued is not None:
        normalized_biblio_record.dateIssued = normalize_date_issued(biblio_record.dateIssued)

    if biblio_record.manufacturePublisher is not None:
        normalized_biblio_record.manufacturePublisher = normalize_manufacture_publisher(biblio_record.manufacturePublisher)

    if biblio_record.manufacturePlaceTerm is not None:
        normalized_biblio_record.manufacturePlaceTerm = normalize_manufacture_place_term(biblio_record.manufacturePlaceTerm)

    if biblio_record.publisher is not None:
        normalized_publishers = []
        for publisher in biblio_record.publisher:
            normalized_publishers.append(normalize_publisher(publisher))
        normalized_biblio_record.publisher = normalized_publishers

    if biblio_record.author is not None:
        normalized_authors = []
        for author in biblio_record.author:
            normalized_authors.append(normalize_author(author))
        normalized_biblio_record.author = normalized_authors

    if biblio_record.illustrator is not None:
        normalized_illustrators = []
        for illustrator in biblio_record.illustrator:
            normalized_illustrators.append(normalize_illustrator(illustrator))
        normalized_biblio_record.illustrator = normalized_illustrators

    if biblio_record.translator is not None:
        normalized_translators = []
        for translator in biblio_record.translator:
            normalized_translators.append(normalize_translator(translator))
        normalized_biblio_record.translator = normalized_translators

    if biblio_record.editor is not None:
        normalized_editors = []
        for editor in biblio_record.editor:
            normalized_editors.append(normalize_editor(editor))
        normalized_biblio_record.editor = normalized_editors

    return normalized_biblio_record


def normalize_title(title: str):
    return normalize_title_name(title)

def normalize_sub_title(sub_title: str):
    return normalize_title_name(sub_title)

def normalize_part_name(part_name: str):
    return normalize_title_name(part_name)

def normalize_part_number(part_number: str):
    return normalize_number(part_number)

def normalize_series_name(series_name: str):
    return normalize_title_name(series_name)

def normalize_series_number(series_number: str):
    return normalize_number(series_number)

def normalize_edition(edition: str):
    edition = edition.strip()
    edition = remove_trailing_punctuation(edition)
    edition = remove_multiple_whitespaces(edition)
    edition = edition.lower()
    return edition

def normalize_place_term(place_term: str):
    return normalize_place(place_term)

def normalize_date_issued(date_issued: str):
    date_issued = date_issued.strip()
    date_issued = remove_trailing_punctuation(date_issued)
    date_issued = remove_multiple_whitespaces(date_issued)
    date_issued = date_issued.lower()
    return date_issued

def normalize_manufacture_publisher(manufacture_publisher: str):
    return normalize_person_name(manufacture_publisher)

def normalize_manufacture_place_term(manufacture_place_term: str):
    return normalize_place(manufacture_place_term)

def normalize_publisher(publisher: str):
    return normalize_person_name(publisher)

def normalize_author(author: str):
    return normalize_person_name(author)

def normalize_illustrator(illustrator: str):
    return normalize_person_name(illustrator)

def normalize_translator(translator: str):
    return normalize_person_name(translator)

def normalize_editor(editor: str):
    return normalize_person_name(editor)

def normalize_title_name(title_name: str):
    title_name = title_name.strip()
    title_name = remove_multiple_whitespaces(title_name)
    title_name = title_name.lower()
    return title_name

def normalize_number(number: str):
    number = number.strip()
    number = remove_trailing_punctuation(number)
    number = remove_multiple_whitespaces(number)
    number = number.lower()
    return number

def normalize_place(place: str):
    place = place.strip()
    place = remove_trailing_punctuation(place)
    place = remove_multiple_whitespaces(place)
    place = place.lower()
    return place

def normalize_person_name(person_name: str):
    person_name = person_name.strip()
    person_name = remove_trailing_punctuation(person_name)
    person_name = remove_multiple_whitespaces(person_name)
    person_name = person_name.lower()
    return person_name

def remove_trailing_punctuation(text: str):
    # Remove trailing .,;: and all types of dashes
    return re.sub(r"[.,;:‐‑‒–—―]+$", "", text)

def remove_multiple_whitespaces(text: str):
    # Replace multiple whitespace or single non-space whitespace with a single space
    return re.sub(r"\s+", " ", text)


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
