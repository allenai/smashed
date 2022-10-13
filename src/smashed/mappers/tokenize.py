"""


@lucas, @kylel

"""
import unicodedata
from typing import Any, List, Optional

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..base import SingleBaseMapper, TransformElementType


class GetTokenizerOutputFieldsMixin:
    """A mixin class that figures out the output fields based on the arguments
    that will be passed a to tokenizer.__call__ method."""

    tokenizer: PreTrainedTokenizerBase

    def output_fields_from_tokenizer_kwargs(
        self,
        tokenizer_kwargs: Optional[dict] = None
    ) -> List[str]:

        tokenizer_kwargs = tokenizer_kwargs or {}

        output_fields = ['input_ids']

        if 'return_attention_mask' in tokenizer_kwargs:
            output_fields.append("attention_mask")
        if 'return_token_type_ids' in tokenizer_kwargs:
            output_fields.append("token_type_ids")
        if 'return_overflowing_tokens' in tokenizer_kwargs:
            output_fields.append("overflow_to_sample_mapping")
        if 'return_special_tokens_mask' in tokenizer_kwargs:
            output_fields.append("special_tokens_mask")
        if 'return_offsets_mapping' in tokenizer_kwargs:
            output_fields.append("offset_mapping")
        if 'return_length' in tokenizer_kwargs:
            output_fields.append("length")

        return output_fields


class TokenizerMapper(SingleBaseMapper, GetTokenizerOutputFieldsMixin):
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
        **tokenizer_kwargs: Any,
    ) -> None:
        self.prefix = f"{output_prefix}_" if output_prefix else ""
        self.to_tokenize_filed = input_field
        self.tokenizer = tokenizer

        # arguments to be passed to the tokenizer __call__ function go here
        tokenizer_kwargs = {
            "add_special_tokens": add_special_tokens,
            "max_length": max_length,
            "is_split_into_words": is_split_into_words,
            "return_attention_mask": return_attention_mask,
            "return_token_type_ids": return_token_type_ids,
            "return_overflowing_tokens": return_overflowing_tokens,
            "return_special_tokens_mask": return_special_tokens_mask,
            "return_offsets_mapping": return_offsets_mapping,
            "return_length": return_length,
            **(tokenizer_kwargs or {}),
        }

        output_fields = self.output_fields_from_tokenizer_kwargs(
            tokenizer_kwargs=tokenizer_kwargs
        )

        # beside the fields returned by the tokenizer, we might also want
        # to return the word ids and the words themselves, depending on
        # options provided.
        self.return_word_ids = self.return_words = False
        if "is_split_into_words" in tokenizer_kwargs and return_word_ids:
            self.return_word_ids = True
            output_fields.append("word_ids")
            if return_words:
                self.return_words = True
                output_fields.append("words")

        self.tokenize_kwargs = tokenizer_kwargs

        super().__init__(
            input_fields=[self.to_tokenize_filed],
            output_fields=list(map(self._prefixify, output_fields)),
        )

    def _prefixify(self, field_or_dict: str) -> str:
        return f"{self.prefix}{field_or_dict}"

    def transform(self, data: TransformElementType) -> TransformElementType:
        batch_encoding = self.tokenizer(
            (to_tok_field := data[self.to_tokenize_filed]),
            **self.tokenize_kwargs
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
                        None if wid is None else to_tok_field[wid]
                        for wid in wids
                    ]
                    # ignoring pylance complaining because we just added
                    # word_ids above!
                    for wids in batch_encoding["word_ids"]  # pyright: ignore
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
    ):
        """
        Args:
            input_fields (str): list of fields to be processed
            unicode_categories (List[str]): list of unicode categories to
                replace. See https://fileformat.info/info/unicode/category
                for a list of categories.
            replace_token (str): token to replace invalid unicode characters
                with.
        """

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
        """
        Args:
            pad_to_length (int): length to pad to
            pad_value (Any): value to pad with
            fields_to_pad (List[str], optional): list of fields to pad.
                If None, all fields will be padded. Defaults to None.
        """

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
