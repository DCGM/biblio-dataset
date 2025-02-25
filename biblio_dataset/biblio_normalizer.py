import re
import regex
import unicodedata

from biblio_dataset.create_biblio_dataset import BiblioRecord
from biblio_dataset.biblio_evaluators import BiblioResult

class BiblioNormalizer:
    def __init__(self, replace_special_characters=True, lowercase=False, remove_diacritics=False):
        self.replace_special_characters = replace_special_characters
        self.lowercase = lowercase
        self.remove_diacritics = remove_diacritics

    def normalize_biblio_record(self, biblio_record: BiblioRecord) -> BiblioRecord:
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

    def normalize_biblio_result(self, biblio_result: BiblioResult) -> BiblioResult:
        normalized_biblio_result = BiblioResult(task_id=biblio_result.task_id,
                                                library_id=biblio_result.library_id)

        if biblio_result.title is not None:
            normalized_titles = []
            for title in biblio_result.title:
                normalized_titles.append((self.normalize(title[0], self.normalize_title), title[1]))
            normalized_biblio_result.title = normalized_titles

        if biblio_result.subTitle is not None:
            normalized_sub_titles = []
            for sub_title in biblio_result.subTitle:
                normalized_sub_titles.append((self.normalize(sub_title[0], self.normalize_sub_title), sub_title[1]))
            normalized_biblio_result.subTitle = normalized_sub_titles

        if biblio_result.partName is not None:
            normalized_part_names = []
            for part_name in biblio_result.partName:
                normalized_part_names.append((self.normalize(part_name[0], self.normalize_part_name), part_name[1]))
            normalized_biblio_result.partName = normalized_part_names

        if biblio_result.partNumber is not None:
            normalized_part_numbers = []
            for part_number in biblio_result.partNumber:
                normalized_part_numbers.append(
                    (self.normalize(part_number[0], self.normalize_part_number), part_number[1]))
            normalized_biblio_result.partNumber = normalized_part_numbers

        if biblio_result.seriesName is not None:
            normalized_series_names = []
            for series_name in biblio_result.seriesName:
                normalized_series_names.append(
                    (self.normalize(series_name[0], self.normalize_series_name), series_name[1]))
            normalized_biblio_result.seriesName = normalized_series_names

        if biblio_result.seriesNumber is not None:
            normalized_series_numbers = []
            for series_number in biblio_result.seriesNumber:
                normalized_series_numbers.append(
                    (self.normalize(series_number[0], self.normalize_series_number), series_number[1]))
            normalized_biblio_result.seriesNumber = normalized_series_numbers

        if biblio_result.edition is not None:
            normalized_editions = []
            for edition in biblio_result.edition:
                normalized_editions.append((self.normalize(edition[0], self.normalize_edition), edition[1]))
            normalized_biblio_result.edition = normalized_editions

        if biblio_result.placeTerm is not None:
            normalized_place_terms = []
            for place_term in biblio_result.placeTerm:
                normalized_place_terms.append((self.normalize(place_term[0], self.normalize_place_term), place_term[1]))
            normalized_biblio_result.placeTerm = normalized_place_terms

        if biblio_result.dateIssued is not None:
            normalized_date_issued = []
            for date_issued in biblio_result.dateIssued:
                normalized_date_issued.append((self.normalize(date_issued[0], self.normalize_date_issued), date_issued[1]))
            normalized_biblio_result.dateIssued = normalized_date_issued

        if biblio_result.manufacturePublisher is not None:
            normalized_manufacture_publishers = []
            for manufacture_publisher in biblio_result.manufacturePublisher:
                normalized_manufacture_publishers.append((self.normalize(
                    manufacture_publisher[0], self.normalize_manufacture_publisher), manufacture_publisher[1]))
            normalized_biblio_result.manufacturePublisher = normalized_manufacture_publishers

        if biblio_result.manufacturePlaceTerm is not None:
            normalized_manufacture_place_terms = []
            for manufacture_place_term in biblio_result.manufacturePlaceTerm:
                normalized_manufacture_place_terms.append((self.normalize(
                    manufacture_place_term[0], self.normalize_manufacture_place_term), manufacture_place_term[1]))
            normalized_biblio_result.manufacturePlaceTerm = normalized_manufacture_place_terms

        if biblio_result.publisher is not None:
            normalized_publishers = []
            for publisher in biblio_result.publisher:
                normalized_publishers.append((self.normalize(publisher[0], self.normalize_publisher), publisher[1]))
            normalized_biblio_result.publisher = normalized_publishers

        if biblio_result.author is not None:
            normalized_authors = []
            for author in biblio_result.author:
                normalized_authors.append((self.normalize(author[0], self.normalize_author), author[1]))
            normalized_biblio_result.author = normalized_authors

        if biblio_result.illustrator is not None:
            normalized_illustrators = []
            for illustrator in biblio_result.illustrator:
                normalized_illustrators.append(
                    (self.normalize(illustrator[0], self.normalize_illustrator), illustrator[1]))
            normalized_biblio_result.illustrator = normalized_illustrators

        if biblio_result.translator is not None:
            normalized_translators = []
            for translator in biblio_result.translator:
                normalized_translators.append((self.normalize(translator[0], self.normalize_translator), translator[1]))
            normalized_biblio_result.translator = normalized_translators

        if biblio_result.editor is not None:
            normalized_editors = []
            for editor in biblio_result.editor:
                normalized_editors.append((self.normalize(editor[0], self.normalize_editor), editor[1]))
            normalized_biblio_result.editor = normalized_editors

        return normalized_biblio_result

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