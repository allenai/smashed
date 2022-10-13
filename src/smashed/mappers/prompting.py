from bisect import bisect_right
from dataclasses import dataclass
from enum import Enum
from math import floor
import random
import unicodedata
from typing import Any, List, Literal, Optional, Sequence, Union, cast
from string import Formatter


from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..base import SingleBaseMapper, TransformElementType



@dataclass
class PromptSegment:
    __slots__ = ("prompt_text", "field_name", "prompt_token_ids",)

    prompt_text: str
    field_name: Union[str, None]
    prompt_token_ids: List[int]
    truncate: bool

    @classmethod
    def _from_template_single(
        cls,
        literal_text: str,
        field_name: Union[str, None],
        format_spec: Union[str, None],
        tokenizer: PreTrainedTokenizerBase,
    ) -> "PromptSegment":
        token_ids = tokenizer.encode(literal_text, add_special_tokens=False)
        return cls(
            prompt_text=literal_text,
            field_name=field_name,
            prompt_token_ids=token_ids,
            truncate=bool(format_spec),
        )

    @classmethod
    def from_template(
        cls, template: str, tokenizer: PreTrainedTokenizerBase,
    ) -> List['PromptSegment']:
        return [
            cls._from_template_single(
                literal_text=literal_text,
                field_name=field_name,
                tokenizer=tokenizer,
                format_spec=format_spec,
            )
            for literal_text, field_name, format_spec, _
            in Formatter().parse(template)
        ]

    def __len__(self):
        return len(self.prompt_token_ids)


class EncodeFieldsMapper(SingleBaseMapper):
    def __init__(
        self,
        fields_to_encode: List[str],
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.tokenizer = tokenizer
        self.fields_to_encode = set(fields_to_encode)
        super().__init__(
            input_fields=fields_to_encode,
            output_fields=fields_to_encode,
        )

    def transform(self, data: TransformElementType) -> TransformElementType:
        tokenized = {
            name: (
                self.tokenizer(
                    field,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                ).input_ids if field in self.fields_to_encode else field
            )
            for name, field in data.items()
        }
        return tokenized


class TruncateFieldsMapper(SingleBaseMapper):
    def __init__(
        self,
        fields_to_truncate: List[str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: Optional[int] = None,
        penalty: int = 0,
        strategy: Union[
            Literal['longest'],
            Literal['uniform'],
         ] = 'longest',
    ):
        max_length = max_length or tokenizer.model_max_length
        if not isinstance(max_length, int):
            raise ValueError(
                f"max_length must be an integer, not {type(max_length)}"
            )

        if strategy not in ('longest', 'uniform'):
            raise ValueError(
                "strategy must be one of 'longest' or 'uniform', "
                f"not {strategy}"
            )

        self.tokenizer = tokenizer
        self.fields_to_truncate = set(fields_to_truncate)
        self.max_length = max_length - penalty
        self.strategy = strategy
        super().__init__(
            input_fields=fields_to_truncate,
            output_fields=fields_to_truncate,
        )

    @classmethod
    def _find_truncated_lens_uniform(
        cls, lens: List[int], max_len: int
    ) -> List[int]:

        reduction_fraction = max_len / sum(lens)

        if reduction_fraction >= 1:
            # no need to cut
            return lens

        # we will cut all sequences by the same amount
        truncated = [floor(ls * reduction_fraction) for ls in lens]
        return truncated

    @classmethod
    def _find_truncated_lens_longest(
        cls, lens: List[int], max_len: int
    ) -> List[int]:
        """Cuts longest sequences first until we get a set of sequences
        that is up to the maximum length."""

        if sum(lens) <= max_len:
            return lens

        # this is how long sequences will be on average.
        target_length = max_len // len(lens)

        # these are the lengths that are above the target length;
        # we need to figure out how much to cut them by by "redistributing"
        # the length from (a) sequences that are below the target length
        # and (b) any rounding errors from // above.
        longer_than_average = [
            ls - target_length if ls > target_length else 0
            for ls in lens
        ]
        extra_len_to_redistribute = (
            # we need to redistribute what we have lost in rounding...
            max_len - target_length * len(lens)
            # ... plus the extra saving
            + sum(target_length - ls for ls in lens if ls < target_length)
        )

        # we actually leverage _find_truncated_lens_uniform to calculate
        # how much to redistribute to each sequence above the target length
        redistributed_extra_len = cls._find_truncated_lens_uniform(
            lens=longer_than_average,
            max_len=extra_len_to_redistribute,
            # max_length=max_len,
        )

        # we figure out new lengths by adding the redistributed extra length
        # to lengths above the target length, and keeping the one below target
        # length as is.
        return [
            target_length + le if ls > target_length else ls
            for ls, le in zip(lens, redistributed_extra_len)
        ]

    def transform(self, data: TransformElementType) -> TransformElementType:
        sequences_to_truncate = [
            field for name, field in data.items()
            if name in self.fields_to_truncate
        ]

        # total_original_len = sum(lens)



# class TruncateFieldsInPrompt(SingleBaseMapper, GetTokenizerOutputFieldsMixin):
#     def __init__(
#         self,
#         template: str,
#         tokenizer: PreTrainedTokenizerBase,
#         max_length: Optional[int] = None,
#         truncation_strategy: str = "longest",
#         **tk_kwargs: Any,
#     ) -> None:
#         self.tokenizer = tokenizer

#         self.prompt = PromptSegment.from_template(
#             template=template,
#             tokenizer=tokenizer
#         )
#         self.max_length = (
#             max_length or tokenizer.model_max_length
#         ) - sum(len(p) for p in self.prompt)

#         output_fields = self.get_tokenizer_output_fields(**tk_kwargs)
#         super().__init__(
#             input_fields=[ps.field_name for ps in self.prompt if ps.field_name],
#             output_fields=output_fields
#         )

#     def transform(self, data: TransformElementType) -> TransformElementType:
#         ...
