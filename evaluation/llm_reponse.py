from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GecError:
    edit: str
    error: str
    error_type: str


@dataclass
class Token:
    start_index: int
    end_index: int
    word: str
    pos_tag: Optional[str] = None
    ner_tag: Optional[str] = None
    gec_error: Optional[GecError] = None

    def to_str(self):
        mapping = {
            'gec': lambda x: f'<g ed="{self.gec_error.edit}" et="{self.gec_error.error_type}">{x}</g>',
            'ner': lambda x: f'<n t="{self.ner_tag}">{x}</n>',
            'pos': lambda x: f'<p t="{self.pos_tag}">{x}</p>',
        }

        word = self.word
        for k, v in [(self.pos_tag, 'pos'), (self.ner_tag, 'ner'), (self.gec_error, 'gec')]:
            if k is not None:
                word = mapping[v](word)
        return word


@dataclass
class TextData:
    text: str
    tokens: List[Token]
