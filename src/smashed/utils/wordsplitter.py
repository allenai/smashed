from typing import List, Sequence, Union

from necessary import Necessary, necessary
from tokenizers.pre_tokenizers import Whitespace, WhitespaceSplit

with necessary("blingfire", soft=True) as BLINGFIRE_AVAILABLE:
    if BLINGFIRE_AVAILABLE:
        from blingfire import text_to_words


__all__ = ["WhitespaceSplitter", "BlingFireSplitter"]


class BaseWordSplitter:
    def __init__(self, language: str = "en"):
        self.language = language

    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError()

    def __call__(
        self, text: Union[str, Sequence[str]]
    ) -> Union[List[str], List[List[str]]]:
        if isinstance(text, str):
            return self.tokenize(text)
        else:
            return [self.tokenize(t) for t in text]


@Necessary(
    "blingfire",
    message="{module_name} missing. Fix with 'pip install smashed[prompting]'",
)
class BlingFireSplitter(BaseWordSplitter):
    def tokenize(self, text: str) -> List[str]:
        return text_to_words(text).split()


class WhitespaceSplitter(BaseWordSplitter):
    def __init__(self, language: str = "en"):
        super().__init__(language)
        self.tokenizer = WhitespaceSplit()

    def tokenize(self, text: str) -> List[str]:
        return [e for e, _ in self.tokenizer.pre_tokenize_str(text)]


class WhitespacePlusSplitter(WhitespaceSplitter):
    def __init__(self, language: str = "en"):
        super().__init__(language)
        self.tokenizer = Whitespace()
