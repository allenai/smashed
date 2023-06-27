from typing import TYPE_CHECKING, List, Sequence, Union

from necessary import Necessary, necessary

with necessary(
    "blingfire", soft=True, errors=(ModuleNotFoundError, OSError)
) as BLINGFIRE_AVAILABLE:
    if BLINGFIRE_AVAILABLE or TYPE_CHECKING:
        from blingfire import text_to_words


with necessary("tokenizers", soft=True) as TOKENIZER_AVAILABLE:
    if TOKENIZER_AVAILABLE or TYPE_CHECKING:
        from tokenizers.pre_tokenizers import Whitespace, WhitespaceSplit


__all__ = [
    "BlingFireSplitter",
    "TrivialTokenizer",
    "WhitespacePlusSplitter",
    "WhitespaceSplitter",
    "WhitespaceTrailSplitter",
]


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
    errors=(ModuleNotFoundError, OSError),
    message=(
        "{module_name} missing. Fix with 'pip install smashed[prompting]'."
        "If you are on a Mac with Apple Silicon chip, also run "
        "'python -m smashed.utils.install_blingfire_macos' before pip."
    ),
)
class BlingFireSplitter(BaseWordSplitter):
    def tokenize(self, text: str) -> List[str]:
        return text_to_words(text).split()


@Necessary(
    "tokenizers",
    message="{module_name} missing. Run 'pip install smashed[prompting]'",
)
class WhitespaceSplitter(BaseWordSplitter):
    def __init__(self, language: str = "en"):
        super().__init__(language)
        self.tokenizer = WhitespaceSplit()

    def tokenize(self, text: str) -> List[str]:
        return [e for e, _ in self.tokenizer.pre_tokenize_str(text)]


@Necessary(
    "tokenizers",
    message="{module_name} missing. Run 'pip install smashed[prompting]'",
)
class WhitespacePlusSplitter(WhitespaceSplitter):
    def __init__(self, language: str = "en"):
        super().__init__(language)
        self.tokenizer = Whitespace()


@Necessary(
    "tokenizers",
    message="{module_name} missing. Run 'pip install smashed[prompting]'",
)
class WhitespaceTrailSplitter(WhitespacePlusSplitter):
    def tokenize(self, text: str) -> List[str]:
        # the start of each token
        locs = [s for _, (s, _) in self.tokenizer.pre_tokenize_str(text)]

        # we include any trailing whitespace in the token
        return [text[locs[i] : locs[i + 1]] for i in range(len(locs) - 1)] + [
            text[locs[-1] :]
        ]


class TrivialTokenizer(BaseWordSplitter):
    def tokenize(self, text: str) -> List[str]:
        return text.strip().split()
