"""
Bunch of decoding mappers to reverse tokenization

@lucas
"""

from typing import Any, Dict, Optional, Sequence, Union

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..base import SingleBaseMapper, TransformElementType

__all__ = ["DecodingMapper"]


class DecodingMapper(SingleBaseMapper):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        fields: Union[str, Sequence[str]],
        decode_batch: bool = False,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        extra_decode_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """A mapper that decodes one or more of tokenized sequences in
        for the provided fields.

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to use for
                decoding; typically, this is the same tokenizer that was used
                for tokenization.
            fields (Union[str, Sequence[str]]): The fields to decode; could
                either be a single field or a sequence of fields.
            decode_batch (bool, optional): If True, it assume each sample is
                a sequence of sequences to decode and will use the tokenizer's
                `batch_decode` method. If False, it assume each sample contains
                a single sequence to decode and will use the tokenizer's
                `decode` method. Defaults to False.
            skip_special_tokens (bool, optional): Whether to skip special
                tokens (e.g., `[CLS]`, `</>`, etc) when decoding. Defaults to
                False.
            clean_up_tokenization_spaces (bool, optional): Whether to clean
                up redundant spaces when decoding. Defaults to True.
            extra_decode_kwargs (Optional[Dict[str, Any]], optional): Any
                tokenizer-specific arguments to pass to the tokenizer's
                `batch_decode` method. If not provided, no extra arguments
                will be passed. Defaults to None.
        """

        self.tokenizer = tokenizer
        self.fields = [fields] if isinstance(fields, str) else fields
        self.decode_batch = decode_batch
        self.skip_special_tokens = skip_special_tokens
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        self.extra_decode_kwargs = extra_decode_kwargs or {}
        super().__init__(input_fields=self.fields, output_fields=self.fields)

    def transform(self, data: TransformElementType) -> TransformElementType:
        return {
            field: (
                self.tokenizer.batch_decode
                if self.decode_batch
                else self.tokenizer.decode
            )(
                data[field],
                skip_special_tokens=self.skip_special_tokens,
                clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
                **self.extra_decode_kwargs,
            )
            for field in self.fields
        }
