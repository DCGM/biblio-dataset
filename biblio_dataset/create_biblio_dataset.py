import argparse
import json
import logging
import os
import re
import regex
import sys
import unicodedata

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
        biblio_record_normalizer = BiblioRecordNormalizer(lowercase=args.lowercase, remove_diacritics=args.remove_diacritics)
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


class BiblioRecordNormalizer:
    def __init__(self, replace_special_characters=True, lowercase=False, remove_diacritics=False):
        self.replace_special_characters = replace_special_characters
        self.lowercase = lowercase
        self.remove_diacritics = remove_diacritics

    def normalize_biblio_record(self, biblio_record: BiblioRecord):
        normalized_biblio_record = BiblioRecord(task_id=biblio_record.task_id,
                                                library_id=biblio_record.library_id)

        if biblio_record.title is not None:
            normalized_biblio_record.title = self.normalize(biblio_record.title, self.normalize_title)

        if biblio_record.subTitle is not None:
            normalized_biblio_record.subTitle = self.normalize(biblio_record.subTitle, self.normalize_sub_title)

        if biblio_record.partName is not None:
            normalized_biblio_record.partName = self.normalize(biblio_record.partName, self.normalize_part_name)

        if biblio_record.partNumber is not None:
            normalized_biblio_record.partNumber = self.normalize(biblio_record.partNumber, self.normalize_part_number)

        if biblio_record.seriesName is not None:
            normalized_biblio_record.seriesName = self.normalize(biblio_record.seriesName, self.normalize_series_name)

        if biblio_record.seriesNumber is not None:
            normalized_biblio_record.seriesNumber = self.normalize(biblio_record.seriesNumber, self.normalize_series_number)

        if biblio_record.edition is not None:
            normalized_biblio_record.edition = self.normalize(biblio_record.edition, self.normalize_edition)

        if biblio_record.placeTerm is not None:
            normalized_biblio_record.placeTerm = self.normalize(biblio_record.placeTerm, self.normalize_place_term)

        if biblio_record.dateIssued is not None:
            normalized_biblio_record.dateIssued = self.normalize(biblio_record.dateIssued, self.normalize_date_issued)

        if biblio_record.manufacturePublisher is not None:
            normalized_biblio_record.manufacturePublisher = self.normalize(biblio_record.manufacturePublisher, self.normalize_manufacture_publisher)

        if biblio_record.manufacturePlaceTerm is not None:
            normalized_biblio_record.manufacturePlaceTerm = self.normalize(biblio_record.manufacturePlaceTerm, self.normalize_manufacture_place_term)

        if biblio_record.publisher is not None:
            normalized_publishers = []
            for publisher in biblio_record.publisher:
                normalized_publishers.append(self.normalize(publisher, self.normalize_publisher))
            normalized_biblio_record.publisher = normalized_publishers

        if biblio_record.author is not None:
            normalized_authors = []
            for author in biblio_record.author:
                normalized_authors.append(self.normalize(author, self.normalize_author))
            normalized_biblio_record.author = normalized_authors

        if biblio_record.illustrator is not None:
            normalized_illustrators = []
            for illustrator in biblio_record.illustrator:
                normalized_illustrators.append(self.normalize(illustrator, self.normalize_illustrator))
            normalized_biblio_record.illustrator = normalized_illustrators

        if biblio_record.translator is not None:
            normalized_translators = []
            for translator in biblio_record.translator:
                normalized_translators.append(self.normalize(translator, self.normalize_translator))
            normalized_biblio_record.translator = normalized_translators

        if biblio_record.editor is not None:
            normalized_editors = []
            for editor in biblio_record.editor:
                normalized_editors.append(self.normalize(editor, self.normalize_editor))
            normalized_biblio_record.editor = normalized_editors

        return normalized_biblio_record

    def normalize(self, text: str, attribute_normalization_func):
        if self.replace_special_characters:
            text = self.replace_special_characters_func(text)

        text = attribute_normalization_func(text)

        if self.lowercase:
            text = text.lower()
        if self.remove_diacritics:
            text = self.remove_diacritics_func(text)

        text = text.strip()
        text = self.remove_multiple_whitespaces(text)
        return text


    def normalize_title(self, title: str):
        return self.normalize_title_name(title)

    def normalize_sub_title(self, sub_title: str):
        return self.normalize_title_name(sub_title)

    def normalize_part_name(self, part_name: str):
        return self.normalize_title_name(part_name)

    def normalize_part_number(self, part_number: str):
        return self.normalize_number(part_number)

    def normalize_series_name(self, series_name: str):
        return self.normalize_title_name(series_name)

    def normalize_series_number(self, series_number: str):
        return self.normalize_number(series_number)

    def normalize_edition(self, edition: str):
        edition = self.strip_punctation(edition)
        edition = self.remove_qoutes(edition)
        return edition

    def normalize_place_term(self, place_term: str):
        return self.normalize_place(place_term)

    def normalize_date_issued(self, date_issued: str):
        date_issued = self.strip_punctation(date_issued)
        date_issued = self.remove_qoutes(date_issued)
        return date_issued

    def normalize_manufacture_publisher(self, manufacture_publisher: str):
        return self.normalize_person_name(manufacture_publisher)

    def normalize_manufacture_place_term(self, manufacture_place_term: str):
        return self.normalize_place(manufacture_place_term)

    def normalize_publisher(self, publisher: str):
        return self.normalize_person_name(publisher)

    def normalize_author(self, author: str):
        return self.normalize_person_name(author)

    def normalize_illustrator(self, illustrator: str):
        return self.normalize_person_name(illustrator)

    def normalize_translator(self, translator: str):
        return self.normalize_person_name(translator)

    def normalize_editor(self, editor: str):
        return self.normalize_person_name(editor)

    def normalize_title_name(self, title_name: str):
        return title_name

    def normalize_number(self, number: str):
        number = self.strip_punctation(number)
        number = self.remove_qoutes(number)
        return number

    def normalize_place(self, place: str):
        place = self.strip_punctation(place)
        place = self.remove_qoutes(place)
        return place

    def normalize_person_name(self, person_name: str):
        person_name = self.strip_punctation(person_name)
        person_name = self.remove_qoutes(person_name)
        return person_name

    def strip_punctation(self, text: str):
        # Remove trailing .,;: and all types of dashes
        return regex.sub(r"^\p{P}+|\p{P}+$", "", text)

    def remove_qoutes(self, text: str):
        return re.sub(r"[‘’“”'\"`„«»‹›〈〉″]", "", text)

    def remove_multiple_whitespaces(self, text: str):
        # Replace multiple whitespace or single non-space whitespace with a single space
        return re.sub(r"\s+", " ", text)

    def remove_diacritics_func(self, text: str):
        return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))

    def replace_special_characters_func(self, text: str):
        #.: 9275
        #,: 2038
        #-: 720
        #ſ: 441
        text = text.replace('ſ', 's')
        #:: 410
        #„: 214
        text = text.replace('„', '"')
        #“: 212
        text = text.replace('“', '"')
        #): 174
        #(: 168
        #æ: 161
        text = text.replace('æ', 'ae')
        #/: 154
        #ü: 112
        #ö: 100
        #Æ: 97
        text = text.replace('Æ', 'AE')
        #&: 83
        #»: 83
        text = text.replace('»', '')
        #«: 79
        text = text.replace('«', '')
        #;: 79
        #—: 60
        text = re.sub(r"[‐‑‒–—―]", "-", text)
        #ä: 56
        #': 48
        #=: 37
        #Ö: 31
        #ß: 25
        #ľ: 24
        text = text.replace('ľ', 'l')
        #?: 20
        #â: 18
        #Ü: 18
        #!: 15
        #ꝛ: 13
        text = text.replace('ꝛ', 'r')
        #è: 11
        #Ä: 11
        #à: 10
        #: 6
        #ù: 6
        #ô: 6
        #œ: 5
        text = text.replace('œ', 'oe')
        #ç: 5
        text = text.replace('ç', 'c')
        #ë: 4
        #": 4
        #ł: 4
        text = text.replace('ł', 'l')
        #�: 4
        text = text.replace('�', '')
        #[: 3
        #]: 3
        #ū: 3
        #ꝙ: 3
        text = text.replace('ꝙ', 'q')
        #È: 2
        #ć: 2
        #*: 2
        text = text.replace('*', '')
        #û: 2
        #Ł: 2
        text = text.replace('Ł', 'L')
        #Ę: 2
        text = text.replace('Ę', 'E')
        #ò: 2
        #И: 2
        #ą: 1
        text = text.replace('ą', 'a')
        #·: 1
        text = text.replace('·', '')
        #+: 1
        text = text.replace('+', '')
        #ũ: 1
        #ê: 1
        #ñ: 1
        text = text.replace('ñ', 'n')
        #Ç: 1
        text = text.replace('Ç', 'C')
        #Ą: 1
        text = text.replace('Ą', 'A')
        #ō: 1
        text = text.replace('ō', 'o')
        #б: 1
        #ṡ: 1
        text = text.replace('ṡ', 's')
        #<: 1
        text = text.replace('<', '')
        #ż: 1
        text = text.replace('ż', 'z')
        #ē: 1
        text = text.replace('ē', 'e')
        #Х: 1
        #О: 1
        #З: 1
        #Я: 1
        #Н: 1
        #Б: 1
        #Л: 1

        #OCR was trained on some sequences that contained &amp;
        text = text.replace('&amp;', '&')

        #Old umlaut characters
        replacements = {
            r'Aͤ': 'Ä',
            r'Eͤ': 'Ë',
            r'Iͤ': 'Ï',
            r'Oͤ': 'Ö',
            r'Uͤ': 'Ü',
            r'aͤ': 'ä',
            r'eͤ': 'ë',
            r'iͤ': 'ï',
            r'oͤ': 'ö',
            r'uͤ': 'ü'
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)

        return text


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
