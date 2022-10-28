from typing import List

from necessary import Necessary, necessary
from tokenizers.pre_tokenizers import Whitespace

with necessary("blingfire", soft=True) as BLINGFIRE_AVAILABLE:
    if BLINGFIRE_AVAILABLE:
        from blingfire import text_to_words


__all__ = ["WhitespaceSplitter", "BlingFireSplitter"]


BLINGFIRE_MESSAGE = ()


class BaseWordSplitter:
    def __init__(self, language: str = "en"):
        self.language = language

    def __call__(self, text: str) -> List[str]:
        raise NotImplementedError()


@Necessary(
    "blingfire",
    message="{module_name} missing. Fix with 'pip install smashed[prompting]'",
)
class BlingFireSplitter(BaseWordSplitter):
    def __call__(self, text: str) -> List[str]:
        return text_to_words(text).split()


class WhitespaceSplitter(BaseWordSplitter):
    def __init__(self, language: str = "en"):
        super().__init__(language)
        self.tokenizer = Whitespace()

    def __call__(self, text: str) -> List[str]:
        return [e for e, _ in self.tokenizer.pre_tokenize_str(text)]
