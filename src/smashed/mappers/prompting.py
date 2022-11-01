import sys
from dataclasses import dataclass
from math import floor
from string import Formatter
from typing import Dict, List, Literal, Optional, Sequence, Union

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from ..base import SingleBaseMapper, TransformElementType
from .tokenize import GetTokenizerOutputFieldsMixin

__all__ = [
    "EncodeFieldsMapper",
    "FillEncodedPromptMapper",
    "FillTextPromptMapper",
    "TruncateMultipleFieldsMapper",
]


class EncodeFieldsMapper(SingleBaseMapper):
    """Simply encodes the fields in the input data using the tokenizer."""

    tokenizer: PreTrainedTokenizerBase
    is_split_into_words: bool
    fields_to_encode: Dict[str, None]

    # We use this as maximum length for the tokenizer in case we are not
    # truncating; we need this otherwise huggingface prints a warning.
    # Using sys.maxsize from here: https://stackoverflow.com/a/7604981/938048
    INT_MAX_LENGTH: int = sys.maxsize

    def __init__(
        self,
        fields_to_encode: Sequence[str],
        tokenizer: PreTrainedTokenizerBase,
        is_split_into_words: bool = False,
        fields_to_return_offset_mapping: Union[Sequence[str], bool] = False,
        offset_prefix: str = "offset",
    ):
        """
        Args:
            fields_to_encode (List[str]): The name of the fields to encode.
            tokenizer (PreTrainedTokenizerBase): A huggingface/tokenizer
                to use for encoding.
            is_split_into_words (bool, optional): Whether the input fields
                are already split into words. Defaults to False.
            fields_to_return_offset_mapping (Union[Sequence[str], bool],
                optional): The fields to return the offset mapping for.
                If True, offset mapping will be returned for all fields;
                if False, no offset mapping will be returned; if a sequence,
                offset mapping will be returned for the fields in the sequence.
                For each field the offset mapping is requested, two additional
                fields will be returned: one with the start offsets and one
                with the end offsets. The name of these additional fields is
                controlled by the `start_offset_prefix` and `end_offset_prefix`
                arguments. Defaults to False.
            offset_prefix (str, optional): The prefix to use for the
                new field with offsets. Defaults to "pos_start".
        """

        if fields_to_return_offset_mapping and not isinstance(
            tokenizer, PreTrainedTokenizerFast
        ):
            raise TypeError(
                "return_offsets_mapping is only supported for fast tokenizers,"
                " i.e. those that inherit from PreTrainedTokenizerFast."
            )

        if isinstance(fields_to_return_offset_mapping, bool):
            # if user provides true, it means they want to return the
            # offsets mapping for all fields; if it is false, they
            # don't want to return it for any field
            fields_to_return_offset_mapping = (
                fields_to_encode if fields_to_return_offset_mapping else []
            )

        self.tokenizer = tokenizer
        self.is_split_into_words = is_split_into_words
        self.offset_mapping_fields = set(fields_to_return_offset_mapping)
        self.offset_prefix = offset_prefix

        # @soldni: using `dict.fromkeys` in place of `frozenset` to avoid
        # issues with hashability: sets are not guaranteed to have the
        # same hash, which causes issues when trying to cache through
        # huggingface datasets.
        self.fields_to_encode = dict.fromkeys(fields_to_encode)

        output_fields = list(self.fields_to_encode) + [
            f"{self.offset_prefix}_{field}"
            for field in self.offset_mapping_fields
        ]

        super().__init__(
            input_fields=self.fields_to_encode,
            output_fields=output_fields,
        )

    def transform(self, data: TransformElementType) -> TransformElementType:
        updated = {}

        for field in self.fields_to_encode:
            return_offset_for_this_field = field in self.offset_mapping_fields

            batch_encoding = self.tokenizer(
                data[field],
                # we are not really truncating given the value of
                # max length, but we need to pass something to avoid
                # a warning from huggingface
                truncation=True,
                max_length=self.INT_MAX_LENGTH,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                return_offsets_mapping=return_offset_for_this_field,
                is_split_into_words=self.is_split_into_words,
            )

            # this is work the word ids
            updated[field] = batch_encoding.input_ids

            if return_offset_for_this_field:
                # by default these are returned as tuples, but some
                # interfaces, like huggingface, would probably complain
                updated[f"{self.offset_prefix}_{field}"] = [
                    list(e) for e in batch_encoding.offset_mapping
                ]

        return updated


class TruncateMultipleFieldsMapper(SingleBaseMapper):
    """Truncate n encoded sequences (a.k.a. list of integers)
    to a maximum length."""

    def __init__(
        self,
        fields_to_truncate: List[str],
        fields_to_preserve: Optional[List[str]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_length: Optional[int] = None,
        length_penalty: int = 0,
        strategy: Union[Literal["longest"], Literal["uniform"]] = "longest",
    ):
        """
        Args:
            fields_to_truncate (List[str]): list of fields to truncate;
                these fields must be a sequence of tokens or token ids.
            fields_to_preserve (List[str]): list of fields to preserve;
                these are fields that are not truncated, but are used to
                compute the length to truncate fields to.
            tokenizer (PreTrainedTokenizerBase, optional): tokenizer to use
                to compute the maximum length; if not provided, max_length
                must be provided. Defaults to None.
            max_length (int, optional): maximum length to truncate to; if not
                provided, tokenizer must be provided. Defaults to None.
            length_penalty (int, optional): additional length penalty to apply
                to the maximum length. This is useful in cases when truncation
                is being done for prompting, so the length of the prompt itself
                must be taken into account. Defaults to 0.
            strategy (Union[Literal['longest'], Literal['uniform']], optional):
                strategy to use to compute the maximum length. If 'longest',
                longer sequences are truncated first to stay within the maximum
                length. If 'uniform', all sequences are truncated by the same
                amount. Defaults to 'longest'.
        """

        if len(fields_to_truncate) == 0:
            raise ValueError("fields_to_truncate must be non-empty")

        if tokenizer is None and max_length is None:
            raise ValueError("Tokenizer or max_length must be provided.")
        elif max_length is None:
            max_length = getattr(tokenizer, "model_max_length", None)

        if not isinstance(max_length, int):
            raise ValueError(
                f"max_length must be an integer, not {max_length} "
                f"({type(max_length)})."
            )

        if strategy not in ("longest", "uniform"):
            raise ValueError(
                "strategy must be one of 'longest' or 'uniform', "
                f"not {strategy}"
            )

        self.tokenizer = tokenizer
        self.fields_to_truncate = tuple(sorted(set(fields_to_truncate)))
        self.fields_to_preserve = tuple(sorted(set(fields_to_preserve or [])))
        self.max_length = max_length - length_penalty
        self.strategy = strategy
        super().__init__(
            input_fields=self.fields_to_truncate + self.fields_to_preserve,
            output_fields=self.fields_to_truncate + self.fields_to_preserve,
        )

    @classmethod
    def _find_truncated_lens_uniform(
        cls, lens: List[int], max_len: int
    ) -> List[int]:
        """Truncate all sequences by the same proportion to stay within the
        maximum length."""

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
            ls - target_length if ls > target_length else 0 for ls in lens
        ]
        extra_len_to_redistribute = (
            # we need to redistribute what we have lost in rounding...
            max_len
            - target_length * len(lens)
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
        # these are the lengths of the fields that we are maybe truncating
        lens_to_truncate = [
            len(data[field]) for field in self.fields_to_truncate
        ]

        # adjust max length to account for the length of the preserved fields
        max_len = self.max_length - sum(
            len(data[field]) for field in self.fields_to_preserve
        )

        if self.strategy == "uniform":
            truncated_lens = self._find_truncated_lens_uniform(
                lens=lens_to_truncate, max_len=max_len
            )
        elif self.strategy == "longest":
            truncated_lens = self._find_truncated_lens_longest(
                lens=lens_to_truncate, max_len=max_len
            )
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

        # we add all the truncated fields to the output
        output = {
            field: data[field][:truncated_len]
            for field, truncated_len in zip(
                self.fields_to_truncate, truncated_lens
            )
        }

        # we add back to the output the fields that we are not truncating
        output.update({k: data[k] for k in self.fields_to_preserve})

        return output


@dataclass
class PromptSegment:
    """Class to represent a segment of a prompt. Not meant to be used
    directly by smashed users.

    A segment of a prompt is a text prefix and optionally field where
    to insert data. For example, the following template:

        "{a} is a {b} with {c}."

    will result in the following segments:

        PromptSegment(prompt_text="", field_name="a")
        PromptSegment(prompt_text=" is a ", field_name="b")
        PromptSegment(prompt_text=" with ", field_name="c")
        PromptSegment(prompt_text=" .", field_name=None)

    To parse a full template into segments, use the `from_template`
    class method.

    Prompts can be filled with text using the `fill_text` method. If a
    tokenized is provided to `from_template`, one can also fill with token
    ids using the `fill_encoded` method.
    """

    __slots__ = ("prompt_text", "field_name", "prompt_token_ids")

    prompt_text: str
    field_name: Union[str, None]
    prompt_token_ids: Optional[List[int]]

    @classmethod
    def _from_template_single(
        cls,
        literal_text: str,
        field_name: Union[str, None],
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> "PromptSegment":
        if tokenizer is not None:
            token_ids = tokenizer.encode(
                literal_text, add_special_tokens=False
            )
        else:
            token_ids = None
        return cls(
            prompt_text=literal_text,
            field_name=field_name,
            prompt_token_ids=token_ids,
        )

    @classmethod
    def from_template(
        cls,
        template: str,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> List["PromptSegment"]:
        return [
            cls._from_template_single(
                literal_text=lt,
                field_name=fn,
                tokenizer=tokenizer,
            )
            for lt, fn, _, _ in Formatter().parse(template)
        ]

    def __len__(self):
        if self.prompt_token_ids is None:
            return len(self.prompt_text)
        else:
            return len(self.prompt_token_ids)

    def fill_encoded(self, data: TransformElementType) -> List[int]:
        if self.prompt_token_ids is None:
            raise ValueError(
                "Cannot fill encoded prompt segment that was initialized"
                " without a tokenizer."
            )

        if self.field_name:
            return self.prompt_token_ids + data[self.field_name]
        else:
            return self.prompt_token_ids

    def fill_text(self, data: TransformElementType) -> str:
        if self.field_name:
            return self.prompt_text + data[self.field_name]
        else:
            return self.prompt_text


class FillTextPromptMapper(SingleBaseMapper):
    """Fills a prompt template with text fields."""

    def __init__(self, prompt_template: str, output_field_name: str):
        self.prompt = PromptSegment.from_template(template=prompt_template)
        self.output_field_name = output_field_name

        super().__init__(
            input_fields=[p.field_name for p in self.prompt if p.field_name],
            output_fields=[output_field_name],
        )

    def transform(self, data: TransformElementType) -> TransformElementType:
        data[self.output_field_name] = "".join(
            segment.fill_text(data) for segment in self.prompt
        )
        return data


class FillEncodedPromptMapper(SingleBaseMapper, GetTokenizerOutputFieldsMixin):
    def __init__(
        self,
        template: str,
        tokenizer: PreTrainedTokenizerBase,
        output_prefix: Optional[str] = None,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = False,
        add_bos_token: bool = True,
        add_eos_token: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self._prefix = output_prefix

        self.return_attention_mask = return_attention_mask
        self.return_token_type_ids = return_token_type_ids

        self.bos_token_ids = (
            []
            if tokenizer.bos_token_id is None or not add_bos_token
            else [tokenizer.bos_token_id]
        )
        self.eos_token_ids = (
            []
            if tokenizer.eos_token_id is None or not add_eos_token
            else [tokenizer.eos_token_id]
        )

        self.prompt = PromptSegment.from_template(
            template=template, tokenizer=tokenizer
        )

        super().__init__(
            input_fields=[p.field_name for p in self.prompt if p.field_name],
            output_fields=[
                self.prefix(field_name)
                for field_name in self.output_fields_from_tokenizer_kwargs(
                    tokenizer_kwargs={
                        "return_attention_mask": return_attention_mask,
                        "return_token_type_ids": return_token_type_ids,
                    }
                )
            ],
        )

    def transform(self, data: TransformElementType) -> TransformElementType:
        encoded_prompt = (
            self.bos_token_ids
            + sum((ps.fill_encoded(data) for ps in self.prompt), [])
            + self.eos_token_ids
        )

        output = {self.prefix("input_ids"): encoded_prompt}
        if self.return_attention_mask:
            output[self.prefix("attention_mask")] = [1] * len(encoded_prompt)
        if self.return_token_type_ids:
            output[self.prefix("token_type_ids")] = [0] * len(encoded_prompt)

        return output
