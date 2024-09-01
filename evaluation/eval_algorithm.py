from typing import List

from pydantic import BaseModel

from evaluation.annotation import TaggedText


class TaggedTextComparison:
    def __init__(self, ignore_additional_tags: bool = False, max_position_deviation: int = 0,
                 penalty_per_extra_tag: float = 0.1) -> None:
        self.ignore_additional_tags = ignore_additional_tags
        self.max_position_deviation = max_position_deviation
        self.penalty_per_extra_tag = penalty_per_extra_tag

    def _calculate_attr_similarity(self, ref_tag: BaseModel, prop_tag: BaseModel) -> float:
        pos_tag_match = ref_tag.pos_tag == prop_tag.pos_tag
        ner_tag_match = ref_tag.ner_tag == prop_tag.ner_tag
        gec_error_match = ref_tag.gec_error == prop_tag.gec_error

        total_attributes = 3
        matched_attributes = sum([pos_tag_match, ner_tag_match, gec_error_match])

        return matched_attributes / total_attributes

    def _calculate_position_match(self, ref_start: int, ref_end: int, prop_start: int, prop_end: int) -> bool:
        return abs(ref_start - prop_start) <= self.max_position_deviation and abs(
            ref_end - prop_end) <= self.max_position_deviation

    def compute_accuracy_score(self, reference: 'TaggedText', proposed: 'TaggedText') -> float:
        ref_tags: List[BaseModel] = reference.get_tags()
        prop_tags: List[BaseModel] = proposed.get_tags()

        if not self.ignore_additional_tags:
            extra_tags_count = max(0, len(prop_tags) - len(ref_tags))
        else:
            extra_tags_count = 0

        total_score = 0.0
        ref_tag_count = len(ref_tags)

        for ref_tag in ref_tags:
            matching_scores = []
            for prop_tag in prop_tags:
                if self._calculate_position_match(ref_tag.start_index, ref_tag.end_index, prop_tag.start_index,
                                                  prop_tag.end_index):
                    attr_similarity = self._calculate_attr_similarity(ref_tag, prop_tag)
                    match_score = 0.5
                    if attr_similarity == 1.0:
                        match_score += 0.5
                    matching_scores.append(match_score)

            if matching_scores:
                total_score += max(matching_scores)

        mean_accuracy = total_score / ref_tag_count if ref_tag_count > 0 else 0.0

        penalty = self.penalty_per_extra_tag * extra_tags_count
        final_accuracy = max(0.0, mean_accuracy - penalty)

        return final_accuracy


if __name__ == "__main__":
    reference_text = TaggedText("<p t='POS'>some <n t='NER'>text</p> here </n>")
    proposed_text = TaggedText("<p t='POS'>some <n t='NER'>text</p> here </n><p t='ADJ'>extra</p>")

    comparator = TaggedTextComparison(max_position_deviation=1, ignore_additional_tags=False)
    accuracy_score = comparator.compute_accuracy_score(reference_text, proposed_text)
    print(f"Accuracy Score: {accuracy_score:.2f}")
