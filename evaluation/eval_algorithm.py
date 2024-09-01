from evaluation.annotation import TaggedText
from evaluation.llm_reponse import Token


class TaggedTextComparison:
    def __init__(self, max_position_deviation: int = 0, ignore_additional_tags: bool = True):
        """
        Initialize the comparison object.

        :param max_position_deviation: Maximum allowed deviation in character positions when comparing tag locations.
        :param ignore_additional_tags: If True, additional tags in the proposed text are ignored in the accuracy calculation.
        """
        self.max_position_deviation = max_position_deviation
        self.ignore_additional_tags = ignore_additional_tags

    def calculate_tag_similarity(self, reference_tag: Token, proposed_tag: Token) -> float:
        """
        Calculate the similarity score between two tags.

        :param reference_tag: The tag from the reference text.
        :param proposed_tag: The tag from the proposed text.
        :return: A similarity score between 0 and 1.
        """
        score = 0.0
        if reference_tag.pos_tag == proposed_tag.pos_tag or reference_tag.ner_tag == proposed_tag.ner_tag or reference_tag.gec_error == proposed_tag.gec_error:
            score += 0.5
        if reference_tag.pos_tag == proposed_tag.pos_tag:
            score += 0.5
        if reference_tag.ner_tag == proposed_tag.ner_tag:
            score += 0.5
        if reference_tag.gec_error == proposed_tag.gec_error:
            score += 0.5
        return score / 1.5

    def are_locations_similar(self, ref_start: int, ref_end: int, prop_start: int, prop_end: int) -> bool:
        """
        Check if the locations of two tags are similar within an allowed deviation.

        :param ref_start: Start position of the reference tag.
        :param ref_end: End position of the reference tag.
        :param prop_start: Start position of the proposed tag.
        :param prop_end: End position of the proposed tag.
        :return: True if locations are considered similar, False otherwise.
        """
        return abs(ref_start - prop_start) <= self.max_position_deviation and abs(ref_end - prop_end) <= self.max_position_deviation

    def compute_accuracy_score(self, reference_text: TaggedText, proposed_text: TaggedText) -> float:
        """
        Compute the accuracy score between two instances of TaggedText.

        :param reference_text: The reference TaggedText instance.
        :param proposed_text: The proposed TaggedText instance.
        :return: An accuracy score between 0 and 1.
        """
        reference_tags = reference_text.get_tags()
        proposed_tags = proposed_text.get_tags()

        total_similarity_score = 0.0
        if len(reference_tags) == 0:
            return 0.0

        for reference_tag in reference_tags:
            similarity_scores = [
                self.calculate_tag_similarity(reference_tag, proposed_tag)
                for proposed_tag in proposed_tags
                if self.are_locations_similar(reference_tag.start_index, reference_tag.end_index, proposed_tag.start_index, proposed_tag.end_index)
            ]
            if similarity_scores:
                total_similarity_score += sum(similarity_scores) / len(similarity_scores)

        average_similarity_score = total_similarity_score / len(reference_tags)

        if not self.ignore_additional_tags:
            extra_tags_count = len(proposed_tags) - len(reference_tags)
            if extra_tags_count > 0:
                penalty = extra_tags_count / len(proposed_tags)
                average_similarity_score -= penalty

        return max(average_similarity_score, 0.0)


if __name__ == "__main__":
    reference_text = TaggedText("<p t='POS'>some <n t='NER'>text</p> here </n>")
    proposed_text = TaggedText("<p t='POS'>some <n t='NER'>text</p> here </n><p t='ADJ'>extra</p>")

    comparator = TaggedTextComparison(max_position_deviation=1, ignore_additional_tags=False)
    accuracy_score = comparator.compute_accuracy_score(reference_text, proposed_text)
    print(f"Accuracy Score: {accuracy_score:.2f}")
