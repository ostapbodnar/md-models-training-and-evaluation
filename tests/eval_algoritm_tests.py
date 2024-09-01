import pytest

from evaluation.annotation import TaggedText
from evaluation.eval_algorithm import TaggedTextComparison
from evaluation.llm_reponse import Token, GecError
from bs4 import BeautifulSoup
from ua_gec.annotated_text import MutableText

def create_tagged_text(html_text):
    return TaggedText(html_text)

class TestTaggedTextComparison:

    def test_exact_match(self):
        reference_text = create_tagged_text("<p t='POS'>some <n t='NER'>text</p> here </n>")
        proposed_text = create_tagged_text("<p t='POS'>some <n t='NER'>text</p> here </n>")
        comparator = TaggedTextComparison()
        accuracy_score = comparator.compute_accuracy_score(reference_text, proposed_text)
        assert accuracy_score == 1.0, f"Expected 1.0 but got {accuracy_score}"

    def test_partial_match(self):
        reference_text = create_tagged_text("<p t='POS'>some <n t='NER'>text</p> here </n>")
        proposed_text = create_tagged_text("<p t='POS'>some <n t='ORG'>text</p> here </n>")
        comparator = TaggedTextComparison()
        accuracy_score = comparator.compute_accuracy_score(reference_text, proposed_text)
        assert 0.0 < accuracy_score < 1.0, f"Expected partial match score, but got {accuracy_score}"

    def test_extra_tags_ignored(self):
        reference_text = create_tagged_text("<p t='POS'>some <n t='NER'>text</p> here </n>")
        proposed_text = create_tagged_text("<p t='POS'>some <n t='NER'>text</p> here </n><p t='ADJ'>extra</p>")
        comparator = TaggedTextComparison(ignore_additional_tags=True)
        accuracy_score = comparator.compute_accuracy_score(reference_text, proposed_text)
        assert accuracy_score == 1.0, f"Expected 1.0 with ignored extra tags, but got {accuracy_score}"

    def test_extra_tags_not_ignored(self):
        reference_text = create_tagged_text("<p t='POS'>some <n t='NER'>text</p> here </n>")
        proposed_text = create_tagged_text("<p t='POS'>some <n t='NER'>text</p> here </n><p t='ADJ'>extra</p>")
        comparator = TaggedTextComparison(ignore_additional_tags=False)
        accuracy_score = comparator.compute_accuracy_score(reference_text, proposed_text)
        assert accuracy_score < 1.0, f"Expected less than 1.0 with extra tags, but got {accuracy_score}"

    def test_no_match(self):
        reference_text = create_tagged_text("<p t='POS'>some text</p>")
        proposed_text = create_tagged_text("<p t='NER'>different content</p>")
        comparator = TaggedTextComparison()
        accuracy_score = comparator.compute_accuracy_score(reference_text, proposed_text)
        assert accuracy_score == 0.0, f"Expected 0.0 but got {accuracy_score}"

    def test_location_deviation_within_limit(self):
        reference_text = create_tagged_text("<p t='POS'>some <n t='NER'>text</p> here </n>")
        proposed_text = create_tagged_text("<p t='POS'>sme <n t='NER'>text</p> here </n>")
        comparator = TaggedTextComparison(max_position_deviation=2)
        accuracy_score = comparator.compute_accuracy_score(reference_text, proposed_text)
        assert accuracy_score == 1.0, f"Expected 1.0 with location deviation within limit, but got {accuracy_score}"

    def test_location_deviation_exceeds_limit(self):
        reference_text = create_tagged_text("<p t='POS'>some <n t='NER'>text</p> here </n>")
        proposed_text = create_tagged_text("<p t='POS'>some <n t='NER'>text   </p> here  </n>")
        comparator = TaggedTextComparison(max_position_deviation=1)
        accuracy_score = comparator.compute_accuracy_score(reference_text, proposed_text)
        assert accuracy_score < 1.0, f"Expected less than 1.0 with location deviation exceeding limit, but got {accuracy_score}"

    def test_complex_tag_structure_match(self):
        reference_text = create_tagged_text(
            """<p t="ADV">quickly</p> <p t="VERB">ran</p> <p t="NOUN">dog</p>"""
        )
        proposed_text = create_tagged_text(
            """<p t="ADV">quickly</p> <p t="VERB">ran</p> <p t="NOUN">dog</p>"""
        )
        comparator = TaggedTextComparison()
        accuracy_score = comparator.compute_accuracy_score(reference_text, proposed_text)
        assert accuracy_score == 1.0, f"Expected 1.0 with exact complex match, but got {accuracy_score}"

    def test_complex_tag_structure_partial_match(self):
        reference_text = create_tagged_text(
            """<p t="ADV">quickly</p> <p t="VERB">ran</p> <p t="NOUN">dog</p>"""
        )
        proposed_text = create_tagged_text(
            """<p t="ADV">quickly</p> <p t="VERB">ran</p> <p t="ADJ">dog</p>"""
        )
        comparator = TaggedTextComparison()
        accuracy_score = comparator.compute_accuracy_score(reference_text, proposed_text)
        assert 0.0 < accuracy_score < 1.0, f"Expected partial score with complex structure, but got {accuracy_score}"

    def test_empty_reference_text(self):
        reference_text = create_tagged_text("")
        proposed_text = create_tagged_text("<p t='POS'>some <n t='NER'>text</p> here </n>")
        comparator = TaggedTextComparison()
        accuracy_score = comparator.compute_accuracy_score(reference_text, proposed_text)
        assert accuracy_score == 0.0, f"Expected 0.0 with empty reference text, but got {accuracy_score}"

    def test_empty_proposed_text(self):
        reference_text = create_tagged_text("<p t='POS'>some <n t='NER'>text</p> here </n>")
        proposed_text = create_tagged_text("")
        comparator = TaggedTextComparison()
        accuracy_score = comparator.compute_accuracy_score(reference_text, proposed_text)
        assert accuracy_score == 0.0, f"Expected 0.0 with empty proposed text, but got {accuracy_score}"

    def test_extra_nested_tags_ignored(self):
        reference_text = create_tagged_text("<p t='POS'>hello <n t='NER'>world</p></n>")
        proposed_text = create_tagged_text("<p t='POS'>hello <n t='NER'>world</p></n><g ed='extra' et='grammar'>!</g>")
        comparator = TaggedTextComparison(ignore_additional_tags=True)
        accuracy_score = comparator.compute_accuracy_score(reference_text, proposed_text)
        assert accuracy_score == 1.0, f"Expected 1.0 with ignored extra nested tags, but got {accuracy_score}"

    def test_complex_tag_with_deviation_and_extra_tags(self):
        reference_text = create_tagged_text("<p t='ADV'>quickly</p> <p t='VERB'>ran</p> <p t='NOUN'>dog</p>")
        proposed_text = create_tagged_text(
            "<p t='ADV'>quickl</p> <p t='VERB'>ran</p> <p t='NOUN'>dog</p> <p t='ADJ'>extra</p>")
        comparator = TaggedTextComparison(max_position_deviation=1, ignore_additional_tags=False)
        accuracy_score = comparator.compute_accuracy_score(reference_text, proposed_text)
        assert 0.0 < accuracy_score < 1.0, f"Expected a partial score with deviation and extra tags, but got {accuracy_score}"
