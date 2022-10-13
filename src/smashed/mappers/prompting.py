from dataclasses import dataclass
from enum import Enum
import unicodedata
from typing import Any, List, Optional, Sequence, Union
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


class TruncateFieldsInPrompt(SingleBaseMapper, GetTokenizerOutputFieldsMixin):
    def __init__(
        self,
        template: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: Optional[int] = None,
        truncation_strategy: str = "longest_first",
        **tk_kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer

        self.prompt = PromptSegment.from_template(
            template=template,
            tokenizer=tokenizer
        )
        self.max_length = (
            max_length or tokenizer.model_max_length
        ) - sum(len(p) for p in self.prompt)

        output_fields = self.get_tokenizer_output_fields(**tk_kwargs)
        super().__init__(
            input_fields=[ps.field_name for ps in self.prompt if ps.field_name],
            output_fields=output_fields
        )

    def transform(self, data: TransformElementType) -> TransformElementType:
        ...
