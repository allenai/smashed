import inspect
from typing import Any, Dict, List, Literal, Sequence, Union, cast

from ftfy import TextFixerConfig, fix_text

from ..base import SingleBaseMapper, TransformElementType
from ..utils.wordsplitter import (
    BaseWordSplitter,
    BlingFireSplitter,
    WhitespacePlusSplitter,
    WhitespaceSplitter,
)


class FtfyMapper(SingleBaseMapper):
    """Uses ftfy to fix text encoding and general weirdness issues."""

    fields_to_fix: Dict[str, None]

    def __init__(
        self,
        input_fields: Union[str, List[str]],
        **ftfy_kwargs: Any,
    ) -> None:
        """Initialize the mapper.

        Args:
            input_fields (Union[str, List[str]]): The fields to fix;
                if a string, it is assumed to be a single field.
            ftfy_kwargs (Any): Any keyword arguments to use to create a
                ftfy.TextFixerConfig object. See the ftfy documentation
                for more information.
        """

        if isinstance(input_fields, str):
            input_fields = [input_fields]

        # @soldni: using `dict.fromkeys` in place of `frozenset` to avoid
        # issues with hashability: sets are not guaranteed to have the
        # same hash, which causes issues when trying to cache through
        # huggingface datasets.
        self.fields_to_fix = dict.fromkeys(input_fields)

        # check if options for ftfy are valid
        valid_ftfy_args = set(inspect.getfullargspec(TextFixerConfig).args)
        for arg_name in ftfy_kwargs:
            if arg_name not in valid_ftfy_args:
                raise ValueError(
                    f"Invalid argument for ftfy.TextFixerConfig: {arg_name}"
                )
        self.ftfy_config = TextFixerConfig(**ftfy_kwargs)

        super().__init__(input_fields=input_fields, output_fields=input_fields)

    def transform(self, data: TransformElementType) -> TransformElementType:
        return {
            field: (
                fix_text(value, config=self.ftfy_config)
                if field in self.fields_to_fix
                else value
            )
            for field, value in data.items()
        }


class TextToWordsMapper(SingleBaseMapper):
    splitter: BaseWordSplitter

    def __init__(
        self,
        fields: Union[str, Sequence[str]],
        splitter: Literal[
            "blingfire", "whitespace", "whitespace_plus"
        ] = "whitespace",
    ):
        if splitter == "blingfire":
            self.splitter = BlingFireSplitter()
        elif splitter == "whitespace_plus":
            self.splitter = WhitespacePlusSplitter()
        elif splitter == "whitespace":
            self.splitter = WhitespaceSplitter()
        else:
            raise ValueError(f"Unknown splitter: {splitter}")

        fields = [fields] if isinstance(fields, str) else fields

        super().__init__(input_fields=fields, output_fields=fields)

    def transform(self, data: TransformElementType) -> TransformElementType:
        return {
            field: self.splitter(data[field]) for field in self.input_fields
        }


class WordsToTextMapper(SingleBaseMapper):
    def __init__(
        self,
        fields: Union[str, Sequence[str]],
        joiner: str = " ",
    ):
        fields = [fields] if isinstance(fields, str) else fields
        self.joiner = joiner

        super().__init__(input_fields=fields, output_fields=fields)

    def _join(self, words: Union[Sequence[str], Sequence[Sequence[str]]]):
        if isinstance(words[0], str):
            return self.joiner.join(cast(Sequence[str], words))
        else:
            return [self.joiner.join(w) for w in words]

    def transform(self, data: TransformElementType) -> TransformElementType:
        return {field: self._join(data[field]) for field in self.input_fields}
