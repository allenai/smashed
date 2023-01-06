from typing import Literal, Optional, Sequence

from torch._utils import classproperty
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..base.recipes import BaseRecipe
from ..mappers.fields import ChangeFieldsMapper
from ..mappers.prompting import TruncateMultipleFieldsMapper
from ..mappers.promptsource import JinjaPromptsourceMapper
from ..mappers.text import TextToWordsMapper, WordsToTextMapper
from ..mappers.tokenize import TokenizerMapper


class PromptsourceRecipe(BaseRecipe):
    @classproperty
    def always_remove_columns(cls) -> bool:
        return True

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        jinja_template: str,
        max_source_content_length: Optional[int] = None,
        max_target_content_length: Optional[int] = None,
        truncation_strategy: Literal["longest", "uniform"] = "longest",
        use_words: bool = True,
        additional_fields_to_keep: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()

        template_mapper = JinjaPromptsourceMapper(
            jinja=jinja_template,
        )
        src_fields, tgt_fields = template_mapper.approximate_input_fields
        src_text, tgt_text = template_mapper.approximate_prompt_text

        if use_words:
            text_to_words = TextToWordsMapper(
                fields=list(set(src_fields + tgt_fields))
            )
            length_src_prompt = len(text_to_words.splitter(src_text))
            length_tgt_prompt = max(
                len(text_to_words.splitter(t)) for t in tgt_text
            )
            self.chain(text_to_words)
        else:
            length_src_prompt = len(src_text)
            length_tgt_prompt = len(tgt_text)

        if max_source_content_length is not None:
            max_source_content_length -= length_src_prompt

            if max_source_content_length < 1:
                raise ValueError(
                    f"max_source_content_length must be at least equal to "
                    f"the length of the source prompt ({length_src_prompt})!"
                )

            self.chain(
                TruncateMultipleFieldsMapper(
                    fields_to_truncate=src_fields,
                    max_length=max_source_content_length,
                    strategy=truncation_strategy,
                )
            )

        if max_target_content_length is not None:
            max_target_content_length -= length_tgt_prompt
            if max_target_content_length < 1:
                raise ValueError(
                    f"max_target_content_length must be at least equal to "
                    f"the length of the target prompt ({length_tgt_prompt})!"
                )

            self.chain(
                TruncateMultipleFieldsMapper(
                    fields_to_truncate=tgt_fields,
                    max_length=max_target_content_length,
                    strategy=truncation_strategy,
                )
            )

        if use_words:
            self.chain(
                WordsToTextMapper(fields=list(set(src_fields + tgt_fields)))
            )

        self.chain(template_mapper)

        # tokenize source
        self.chain(
            TokenizerMapper(
                tokenizer=tokenizer,
                input_field="source",
                add_special_tokens=False,
                return_attention_mask=True,
                truncation=True,
            )
        )
        keep_fields = ["input_ids", "attention_mask"]

        if template_mapper.has_target:
            # tokenize target
            self.chain(
                TokenizerMapper(
                    tokenizer=tokenizer,
                    input_field="target",
                    output_rename_map={"input_ids": "labels"},
                    add_special_tokens=False,
                    return_attention_mask=False,
                    truncation=True,
                )
            )
            keep_fields.append("labels")

        if additional_fields_to_keep:
            keep_fields.extend(additional_fields_to_keep)

        self.chain(ChangeFieldsMapper(keep_fields=keep_fields))
