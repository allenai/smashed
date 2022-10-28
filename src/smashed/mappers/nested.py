from copy import deepcopy
from functools import partial
from typing import Callable, Dict, Literal, Sequence, Union, Any

from ..base import SingleBaseMapper, TransformElementType
from ..utils import Nested
from ..utils.wordsplitter import (
    BaseWordSplitter,
    BlingFireSplitter,
    WhitespaceSplitter,
)

__all__ = [
    "TextTruncateMapper",
    "WordsTruncateMapper",
]


class BaseNestedMapper(SingleBaseMapper):
    def __init__(
        self,
        nested_fields: Union[Sequence[str], Dict[str, Any]],
        *args,
        **kwargs,
    ):
        if not isinstance(nested_fields, dict):
            nested_fields = dict.fromkeys(nested_fields)

        self.nested_fields = tuple(
            (Nested.from_str(field), self._partial(values))
            for field, values in nested_fields.items()
        )
        io_fields = [str(spec.key[0]) for spec, _ in self.nested_fields]
        super().__init__(input_fields=io_fields, output_fields=io_fields)

    def _partial(self, args_or_kwargs: Any) -> Callable[[Any], Any]:
        raise NotImplementedError

    def transform(self, data: TransformElementType) -> TransformElementType:
        data = deepcopy(data)
        for field, fn in self.nested_fields:
            field.edit(data, fn)  # type: ignore
        return data


class TextTruncateMapper(BaseNestedMapper):
    def __init__(self, nested_fields: Dict[str, int]):
        super().__init__(nested_fields)

    def _truncate(self, data: str, trunc_to: int) -> str:
        return data[:trunc_to]

    def _partial(self, args_or_kwargs: int) -> Callable[[str], str]:
        if not isinstance(args_or_kwargs, int):
            raise ValueError(
                "Expected an integer for truncation length, "
                f"got {args_or_kwargs} ({type(args_or_kwargs)})"
            )

        return partial(self._truncate, trunc_to=args_or_kwargs)


class WordsTruncateMapper(TextTruncateMapper):
    splitter: BaseWordSplitter

    def __init__(
        self,
        nested_fields: Dict[str, int],
        splitter: Literal["blingfire", "whitespace"] = "blingfire",
    ):
        if splitter == "blingfire":
            self.splitter = BlingFireSplitter()
        elif splitter == "whitespace":
            self.splitter = WhitespaceSplitter()
        else:
            raise ValueError(f"Unknown splitter: {splitter}")

        super().__init__(nested_fields)

    def _truncate(self, data: str, trunc_to: int) -> str:
        words = self.splitter(data)
        return " ".join(words[:trunc_to])
