"""
Bunch of decoding mappers to reverse tokenization

@lucas
"""

from typing import Any, Dict, Optional, Sequence, Union

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..base import SingleBaseMapper, TransformElementType

__all__ = [
    "DecodingMapper"
]


class DecodingMapper(SingleBaseMapper):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        fields: Union[str, Sequence[str]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        extra_decode_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.tokenizer = tokenizer
        self.fields = [fields] if isinstance(fields, str) else fields
        self.skip_special_tokens = skip_special_tokens
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        self.extra_decode_kwargs = extra_decode_kwargs or {}
        super().__init__(input_fields=self.fields, output_fields=self.fields)

    def transform(self, data: TransformElementType) -> TransformElementType:
        return {
            field: self.tokenizer.batch_decode(
                data[field],
                skip_special_tokens=self.skip_special_tokens,
                clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
                **self.extra_decode_kwargs,
            )
            for field in self.fields
        }
