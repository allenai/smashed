from typing import Literal, Optional, Sequence, Union

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..mappers.fields import RenameFieldsMapper
from ..mappers.prompting import (
    EncodeFieldsMapper,
    FillEncodedPromptMapper,
    TruncateNFieldsMapper,
)


class PromptingMapperRecipe(EncodeFieldsMapper):
    """A recipe of chained mappers for prompting tasks.

    As input, it takes a dictionary of fields that need to be inserted
    into a prompt. It outputs input_ids, etc.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        source_template: str,
        target_template: Optional[str] = None,
        fields_to_truncate: Optional[Sequence[str]] = None,
        target_output_name: Union[
            Literal["labels"], Literal["decoder_input_ids"]
        ] = "labels",
        is_split_into_words: bool = False,
        max_source_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        strategy: Union[Literal["longest"], Literal["uniform"]] = "longest",
        return_attention_mask: bool = True,
        return_token_type_ids: bool = False,
    ):
        fields_to_truncate = fields_to_truncate or []

        source_prompt_mapper = FillEncodedPromptMapper(
            template=source_template,
            tokenizer=tokenizer,
            output_prefix=None,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
        )
        fields_to_encode = source_prompt_mapper.input_fields

        if target_template is not None:
            target_prompt_mapper = FillEncodedPromptMapper(
                template=target_template,
                tokenizer=tokenizer,
                return_attention_mask=False,
                # we will change this later
                output_prefix="decoder_",
            )
            fields_to_encode += target_prompt_mapper.input_fields
        else:
            target_prompt_mapper = None

        super().__init__(
            fields_to_encode=fields_to_encode,
            tokenizer=tokenizer,
            is_split_into_words=is_split_into_words,
        )

        source_fields_to_truncate, source_fields_to_preserve = [], []
        for field_name in source_prompt_mapper.input_fields:
            if field_name in fields_to_truncate:
                source_fields_to_truncate.append(field_name)
            else:
                source_fields_to_preserve.append(field_name)

        print(source_fields_to_truncate, source_fields_to_preserve)

        if source_fields_to_truncate:
            source_truncation_mapper = TruncateNFieldsMapper(
                fields_to_truncate=source_fields_to_truncate,
                fields_to_preserve=source_fields_to_preserve,
                max_length=max_source_length,
                strategy=strategy,
                tokenizer=tokenizer,
                length_penalty=sum(
                    len(ps) for ps in source_prompt_mapper.prompt
                ),
            )
            self.chain(source_truncation_mapper)

        if target_prompt_mapper:
            target_fields_to_truncate, target_fields_to_preserve = [], []
            for field_name in target_prompt_mapper.input_fields:
                if field_name in fields_to_truncate:
                    target_fields_to_truncate.append(field_name)
                else:
                    target_fields_to_preserve.append(field_name)

            if target_fields_to_truncate:
                target_truncation_mapper = TruncateNFieldsMapper(
                    fields_to_truncate=target_fields_to_truncate,
                    fields_to_preserve=target_fields_to_preserve,
                    max_length=max_target_length or max_source_length,
                    strategy=strategy,
                    tokenizer=tokenizer,
                    length_penalty=sum(
                        len(ps) for ps in target_prompt_mapper.prompt
                    ),
                )
                self.chain(target_truncation_mapper)

        self.chain(source_prompt_mapper)

        rename_fields_map = {k: k for k in source_prompt_mapper.output_fields}

        if target_prompt_mapper:
            self.chain(target_prompt_mapper)

            if target_output_name == "labels":
                rename_fields_map["decoder_input_ids"] = "labels"
            else:
                rename_fields_map["decoder_input_ids"] = "decoder_input_ids"

        self.chain(
            RenameFieldsMapper(
                rename_fields_map=rename_fields_map, remove_rest=True
            )
        )
