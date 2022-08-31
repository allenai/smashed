"""


@lucas, @kylel

"""

import unicodedata
from typing import Any, List, Optional

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..base.mapper import SingleBaseMapper
from ..base.types import TransformElementType


class TokenizerMapper(SingleBaseMapper):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        input_field: str,
        output_prefix: Optional[str] = None,
        add_special_tokens: Optional[bool] = True,
        max_length: Optional[int] = None,
        is_split_into_words: Optional[bool] = False,
        return_token_type_ids: Optional[bool] = False,
        return_attention_mask: Optional[bool] = True,
        return_overflowing_tokens: Optional[bool] = False,
        return_special_tokens_mask: Optional[bool] = False,
        return_offsets_mapping: Optional[bool] = False,
        return_length: Optional[bool] = False,
        return_word_ids: Optional[bool] = False,
        return_words: Optional[bool] = False,
        **tk_kwargs: Any,
    ) -> None:
        self.prefix = f"{output_prefix}_" if output_prefix else ""
        self.to_tokenize_filed = input_field
        self.tokenizer = tokenizer

        # arguments to be passed to the tokenizer __call__ function go here
        tk_kwargs = {
            "add_special_tokens": add_special_tokens,
            "max_length": max_length,
            "is_split_into_words": is_split_into_words,
            **(tk_kwargs or {}),
        }

        output_fields = ["input_ids"]

        # various options for the tokenizer affect which fields are returned
        tk_kwargs["return_attention_mask"] = return_attention_mask
        if return_attention_mask:
            output_fields.append("attention_mask")

        tk_kwargs["return_token_type_ids"] = return_token_type_ids
        if return_token_type_ids:
            output_fields.append("token_type_ids")

        tk_kwargs["return_overflowing_tokens"] = return_overflowing_tokens
        if return_overflowing_tokens:
            output_fields.append("overflow_to_sample_mapping")

        tk_kwargs["return_special_tokens_mask"] = return_special_tokens_mask
        if return_special_tokens_mask:
            output_fields.append("special_tokens_mask")

        tk_kwargs["return_offsets_mapping"] = return_offsets_mapping
        if return_offsets_mapping:
            output_fields.append("offset_mapping")

        tk_kwargs["return_length"] = return_length
        if return_length:
            output_fields.append("length")

        self.return_word_ids = self.return_words = False
        if "is_split_into_words" in tk_kwargs and return_word_ids:
            self.return_word_ids = True
            output_fields.append("word_ids")
            if return_words:
                self.return_words = True
                output_fields.append("words")

        self.tokenize_kwargs = tk_kwargs
        super().__init__(
            input_fields=[self.to_tokenize_filed],
            output_fields=list(map(self._prefixify, output_fields)),
        )

    def _prefixify(self, field_or_dict: str) -> str:
        return f"{self.prefix}{field_or_dict}"

    def transform(self, data: TransformElementType) -> TransformElementType:
        batch_encoding = self.tokenizer(
            data[self.to_tokenize_filed], **self.tokenize_kwargs
        )

        # token_to_word mappings are unfortunately not natively returned by
        # HF.tokenizer; so we need to operate separately on the
        # `batch_encoding` object to get this info.
        # if f'{self.prefix}word_ids' in self.output_fields:
        if self.return_word_ids:
            batch_encoding["word_ids"] = [
                # ignoring pylance complaining because word ids should
                # be provided given that we are passing `is_split_into_words`
                batch_encoding[sequence_id].word_ids  # type: ignore
                for sequence_id in range(
                    # ignoring pylance complaining because input_ids should
                    # always be a list of lists when `is_split_into_words`
                    # is set to True.
                    len(batch_encoding["input_ids"])  # type: ignore
                )
            ]
            if self.return_words:
                batch_encoding["words"] = [
                    [
                        None
                        if word_id is None
                        else data[self.input_fields[0]][word_id]
                        for word_id in word_ids
                    ]
                    # ignoring pylance complaining because we just added
                    # word_ids above!
                    for word_ids in batch_encoding["word_ids"]  # type: ignore
                ]

        return {
            self._prefixify(field_name): field_value
            for field_name, field_value in batch_encoding.items()
        }


class ValidUnicodeMapper(SingleBaseMapper):
    """Given input_fields of type List[str], replaces invalid Unicode
    characters with something else"""

    def __init__(
        self,
        input_fields: List[str],
        unicode_categories: List[str],
        replace_token: str,
        # output_prefix: Optional[str] = None,
    ):
        # self.batched = False
        # self.input_fields = input_fields
        # self.prefix = f'{output_prefix}_' if output_prefix else ''
        # self.output_fields = [
        #     f'{self.prefix}{input_field}'
        #     for input_field in self.input_fields
        # ]
        super().__init__(input_fields=input_fields, output_fields=input_fields)
        self.unicode_categories = unicode_categories
        self.replace_token = replace_token

    def transform(self, data: TransformElementType) -> TransformElementType:
        def _transform(tokens: List[str]) -> List[str]:
            return [
                self.replace_token
                if all(
                    unicodedata.category(ch) in self.unicode_categories
                    for ch in token
                )
                else token
                for token in tokens
            ]

        return {
            k: v if k not in self.input_fields else _transform(v)
            for k, v in data.items()
        }
        # new_data = {f'{self.prefix}{k}': v if k not in self.input_fields
        #             else _transform(v) for k, v in data.items()}
        # return new_data


class PaddingMapper(SingleBaseMapper):
    """Given input_fields of type List[str], figures out how to pad them
    such that all examples in dataset in those fields have same length.
    This can be useful because Huggingface padding has really weird behavior
    for custom keys. For example,

    tokenizer.pad([{'input_ids': [0, 1, 2], 'aaa': [3, 3, 3]},
                   {'input_ids': [3, 4], 'aaa': [4, 4]}])

    will correctly pad `input_ids`, but not `aaa`.  This can break collation
    which often calls `tokenizer.pad`.
    """

    def __init__(
        self,
        pad_to_length: int,
        pad_value: Any,
        fields_to_pad: Optional[List[str]] = None,
    ):
        super().__init__()
        self.pad_to_length = pad_to_length
        self.pad_value = pad_value
        self.fields_to_pad = fields_to_pad

    def transform(self, data: TransformElementType) -> TransformElementType:
        """Add padding to all list elements for the fields we specify."""
        fields_to_pad = (
            data.keys() if self.fields_to_pad is None else self.fields_to_pad
        )

        def _pad(input_elements: List[Any]) -> List[Any]:
            if len(input_elements) > self.pad_to_length:
                raise ValueError(
                    f"PaddingMapper expects every input sequence to be less"
                    f"than or equal to the `pad_to_length`. Please handle"
                    f"any truncation or whatever upstream in a different "
                    f"mapper, such as TokenizerMapper."
                    f"\t{len(input_elements)} > {self.pad_to_length}"
                    f"\t{input_elements}"
                )
            input_elements += [
                self.pad_value
                for _ in range(self.pad_to_length - len(input_elements))
            ]
            return input_elements

        return {
            k: v if k not in fields_to_pad else _pad(v)
            for k, v in data.items()
        }
