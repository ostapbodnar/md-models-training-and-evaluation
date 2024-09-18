from typing import Optional

from bs4 import Tag, BeautifulSoup
from ua_gec.annotated_text import MutableText, _unescape

from evaluation.intersection import has_intersection
from evaluation.llm_reponse import TextData, Token, GecError

# Define color codes
TAG_COLORS = {
    'NOUN': '\033[34m',  # Blue for nouns (calm and neutral)
    'VERB': '\033[32m',  # Green for verbs (natural, action-related)
    'NUMR': '\033[33m',  # Yellow for numerals (attention-grabbing but softer)
    'ADV': '\033[36m',  # Cyan for adverbs (cool and distinct)
    'ADJ': '\033[35m',  # Magenta for adjectives (creative, descriptive)
    'PREP': '\033[37m',  # White for prepositions (neutral, connective)
    'CONJ': '\033[93m',  # Bright yellow for conjunctions (connective, but noticeable)
    'PART': '\033[96m',  # Bright cyan for particles (stands out, but not too harsh)
    'PRON': '\033[95m',  # Bright magenta for pronouns (distinct from adjectives)
    'ADJP': '\033[94m',  # Light blue for participles (softer, less intense)
    'PUNCT': '\033[90m',  # Grey for punctuation (neutral, less noticeable)
    'NER': '\033[42m',  # Bright green for named entities (highlighted, clear)
    'GEC': '\033[41m',  # Red for grammar correction (attention-grabbing)
    'RESET': '\033[0m'  # Reset color to default
}


class TaggedText:
    """Text representation that allows easy replacements and annotations."""

    def __init__(self, text: str) -> None:

        if not isinstance(text, str):
            raise ValueError(f"`text` must be string, not {type(text)}")

        annotations = self._parse(text)
        self._annotations = annotations.tokens
        self._text = annotations.text

    def __str__(self):
        """Pretend to be a normal string. """
        return self.get_tagged_text()

    def __repr__(self):
        return "<AnnotatedText('{}')>".format(self.get_tagged_text())

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        if self._text != other._text:
            return False

        if len(self._annotations) != len(other._annotations):
            return False

        for ann in other._annotations:
            if ann != self.get_tag_at(ann.start, ann.end):
                return False

        return True

    def tag(
            self,
            start,
            end,
            pos_tag: Optional[str] = None,
            ner_tag: Optional[str] = None,
            gec_error: Optional[GecError] = None,
    ):

        """Annotate substring as being corrected."""
        if start > end:
            raise ValueError(
                f"Start position {start} should not be greater than end position {end}"
            )

        bad = self._text[start:end]

        new_ann = Token(start, end, bad, pos_tag, ner_tag, gec_error)

        for ann in self._annotations:
            if ann.start_index == start and ann.end_index == end:
                ann.pos_tag = ann.pos_tag or pos_tag
                ann.ner_tag = ann.ner_tag or ner_tag
                ann.gec_error = ann.gec_error or gec_error
        else:
            self._annotations.append(new_ann)

    def _get_overlaps(self, start, end):
        """Find all annotations that overlap with the given range. """

        res = []
        for ann in self._annotations:
            if has_intersection(ann.start_index, ann.end_index, start, end):
                res.append(ann)

        return res

    def get_tags(self):
        """Return list of all annotations in the text. """
        return self._annotations

    def iter_tags(self):
        """Iterate the annotations in the text."""
        n_anns = len(self._annotations)
        i = 0
        while i < n_anns:
            yield self._annotations[i]
            delta = len(self._annotations) - n_anns
            i += delta + 1
            n_anns = len(self._annotations)

    def get_tag_at(self, start, end=None):
        """Return annotation at the given position or region."""
        if end is None:
            for ann in self._annotations:
                if ann.start_index <= start < ann.end_index:
                    return ann
        else:
            for ann in self._annotations:
                if ann.start_index == start and ann.end_index == end:
                    return ann
        return None

    @staticmethod
    def _parse(text):
        """Return list of annotations found in the text."""
        soup = BeautifulSoup(text, "html.parser")
        parsed_tags = {}

        def recursion(input_tag: Tag, start_index: int):
            raw_sub_text = []
            for tag in input_tag.contents:
                raw_text = "".join(raw_sub_text)
                if isinstance(tag, Tag):
                    index = start_index + len(raw_text)
                    v = recursion(tag, index)
                    word = "".join(v)
                    ident = (index, index + len(word))

                    pos_tag = tag.get("t", "<unk>") if tag.name == "p" else None
                    ner_tag = tag.get("t", "<unk>") if tag.name == "n" else None
                    gec_error = GecError(
                        error=word, edit=tag.get("ed", "<unk>"), error_type=tag.get("et", "<unk>")
                    ) if tag.name == "g" else None

                    if ident in parsed_tags:
                        ann = parsed_tags[ident]
                        ann.pos_tag = ann.pos_tag or pos_tag
                        ann.ner_tag = ann.ner_tag or ner_tag
                        ann.gec_error = ann.gec_error or gec_error
                    else:
                        parsed_tags[ident] = Token(
                            word=word,
                            pos_tag=pos_tag,
                            ner_tag=ner_tag,
                            gec_error=gec_error,
                            start_index=index,
                            end_index=index + len(word)
                        )
                    raw_sub_text.append(word)
                elif isinstance(tag, str):
                    raw_sub_text.append(tag)
                continue

            return raw_sub_text

        result = recursion(soup, 0)
        return TextData(text="".join(result), tokens=list(parsed_tags.values()))

    def remove(self, annotation):
        """Remove annotation, replacing it with the original text."""
        try:
            self._annotations.remove(annotation)
        except ValueError:
            raise ValueError("{} is not in the list".format(annotation))

    def get_original_text(self):
        """Return the original (unannotated) text."""
        return self._text

    def get_tagged_text(self):
        """Return the annotated text."""

        text = MutableText(self._text)
        for ann in self._annotations:
            text.replace(ann.start_index, ann.end_index, ann.to_str())

        return text.get_edited_text()

    def get_corrected_text(self):
        text = MutableText(self._text)
        for ann in self._annotations:
            try:
                text.replace(ann.start_index, ann.end_index, ann.apply_ann())
            except IndexError:
                pass

        return _unescape(text.get_edited_text())

    def get_colored_tagged_text(self):
        """Return the annotated text with colors."""
        text = MutableText(self._text)
        for ann in self._annotations:
            # Determine the color to use: GEC overrides NER and POS, and NER overrides POS
            if ann.gec_error:
                color = TAG_COLORS['GEC']
            elif ann.ner_tag:
                color = TAG_COLORS['NER']
            else:
                color = TAG_COLORS.get(ann.pos_tag, TAG_COLORS['RESET'])

            reset_color = TAG_COLORS['RESET']
            text.replace(ann.start_index, ann.end_index, f"{color}{ann.apply_ann()}{reset_color}")

        return text.get_edited_text()


if __name__ == "__main__":
    # text = TaggedText("<p t='NOUN'>some <n t='NER'>text</p> here </n>")
    # print(text.get_tagged_text())
    # print(text.get_original_text())
    # html_text = """Вчора <p t="ADV">я</p> <p t="PRON">купив</p> <p t="VERB">книгу</p> <p t="NOUN">для</p> <g ed="своїй" et="G/Grammar"><n t="ORG"><p t="ADP">Cвоїй</p></n></g> <p t="NOUN">сестри</p>."""
    html_text = """Вчора <p t="ADV">я</p> <p t="PRON">купив</p> <p t="VERB">книгу</p> <p t="NOUN">для</p> <n t="ORG">Cвоїй</n> <p t="NOUN">сестри</p>."""

    parsed_result = TaggedText(html_text)
    print(parsed_result.get_colored_tagged_text())
