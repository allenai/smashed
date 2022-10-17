from typing import Literal, Optional, Sequence, Union

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..mappers.fields import RenameFieldsMapper
from ..mappers.prompting import (
    EncodeFieldsMapper,
    FillEncodedPromptMapper,
    TruncateNFieldsMapper,
)
from ..mappers.shape import SingleSequenceStriderMapper


class PromptingMapperRecipe(EncodeFieldsMapper):
    """A recipe of chained mappers for prompting tasks.

    As input, it takes a dictionary of fields that need to be inserted
    into a prompt. It outputs input_ids, etc.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        source_template: str,
        source_add_bos_token: bool = True,
        source_add_eos_token: bool = True,
        target_template: Optional[str] = None,
        target_add_bos_token: bool = True,
        target_add_eos_token: bool = True,
        fields_to_truncate: Optional[Sequence[str]] = None,
        fields_to_stride: Optional[Sequence[str]] = None,
        stride_max_length: Optional[int] = None,
        stride_step: Optional[int] = None,
        target_output_name: Union[
            Literal["labels"], Literal["decoder_input_ids"]
        ] = "labels",
        is_split_into_words: bool = False,
        max_source_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        strategy: Union[Literal["longest"], Literal["uniform"]] = "longest",
        return_attention_mask: bool = True,
        return_token_type_ids: bool = False,
        extra_keep_fields: Optional[Sequence[str]] = None,
        extra_encode_fields: Optional[Sequence[str]] = None,
    ):
        fields_to_truncate = fields_to_truncate or []
        fields_to_stride = fields_to_stride or []
        extra_keep_fields = extra_keep_fields or []
        extra_encode_fields = extra_encode_fields or []

        source_prompt_mapper = FillEncodedPromptMapper(
            template=source_template,
            tokenizer=tokenizer,
            output_prefix=None,
            add_bos_token=source_add_bos_token,
            add_eos_token=source_add_eos_token,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
        )
        fields_to_encode = source_prompt_mapper.input_fields + tuple(
            extra_encode_fields or []
        )

        if target_template is not None:
            target_prompt_mapper = FillEncodedPromptMapper(
                template=target_template,
                tokenizer=tokenizer,
                add_bos_token=target_add_bos_token,
                add_eos_token=target_add_eos_token,
                return_attention_mask=False,
                # we will change this later
                output_prefix="decoder",
            )
            fields_to_encode += target_prompt_mapper.input_fields
        else:
            target_prompt_mapper = None

        # this sets up the encoder for all the fields we have to put in
        # the source and target prompts
        super().__init__(
            fields_to_encode=fields_to_encode,
            tokenizer=tokenizer,
            is_split_into_words=is_split_into_words,
        )

        # for the source, this adds truncation / striding depending on
        # whether they are needed or not.
        self._add_truncation_and_striding(
            prompt_mapper=source_prompt_mapper,
            tokenizer=tokenizer,
            all_fields_to_truncate=fields_to_truncate,
            all_fields_to_stride=fields_to_stride,
            strategy=strategy,
            max_length=max_source_length,
            stride_max_length=stride_max_length,
            stride_step=stride_step,
        )

        if target_prompt_mapper:
            # a template for the target sequence has been provided, so
            # we have to truncate and/or stride target fields
            self._add_truncation_and_striding(
                prompt_mapper=target_prompt_mapper,
                tokenizer=tokenizer,
                all_fields_to_truncate=fields_to_truncate,
                all_fields_to_stride=fields_to_stride,
                strategy=strategy,
                max_length=max_target_length or max_source_length,
                stride_max_length=stride_max_length,
                stride_step=stride_step,
            )

        # now that the sequences are ready, we can set up the prompt template
        # filler for the input.
        self.chain(source_prompt_mapper)

        # we need to rename some fields to make sure there is no conflict
        # between the source and target prompts; further this will eliminate
        # all fields that are not needed for this recipe.
        rename_fields_map = {
            k: k
            for k in source_prompt_mapper.output_fields
            + tuple(extra_keep_fields)
        }

        if target_prompt_mapper:

            # we add the target prompt template filler to the chain in case
            # we have a target template
            self.chain(target_prompt_mapper)

            # some models want the target sequence to be called "labels",
            # others want it to be called "decoder_input_ids"; we take care
            # of this here.
            if target_output_name == "labels":
                rename_fields_map["decoder_input_ids"] = "labels"
            else:
                rename_fields_map["decoder_input_ids"] = "decoder_input_ids"

        # finally, this renames the fields to handle aforementioned conflicts
        self.chain(
            RenameFieldsMapper(
                rename_fields_map=rename_fields_map, remove_rest=True
            )
        )

    def _add_truncation_and_striding(
        self,
        prompt_mapper: FillEncodedPromptMapper,
        tokenizer: PreTrainedTokenizerBase,
        all_fields_to_truncate: Sequence[str],
        all_fields_to_stride: Sequence[str],
        strategy: Union[Literal["longest"], Literal["uniform"]],
        max_length: Optional[int] = None,
        stride_max_length: Optional[int] = None,
        stride_step: Optional[int] = None,
    ) -> None:
        fields_to_truncate = []
        fields_to_preserve = []
        fields_to_stride = []

        for field_name in prompt_mapper.input_fields:
            if field_name in all_fields_to_truncate:
                fields_to_truncate.append(field_name)
            else:
                fields_to_preserve.append(field_name)

            if field_name in all_fields_to_stride:
                # note how strided fields still participate into the
                # fields to preserve unless we also want to truncate them
                fields_to_stride.append(field_name)

        # we need to figure out what length we need to stride to; if not
        # provided, we will use the max_length for truncation, which is
        # either max_source_length/max_target_length or the model_max_length
        # from the tokenizer; if none of the three is provided, we will
        # raise an error if there are fields to stride.
        max_length_when_striding = (
            stride_max_length or max_length or tokenizer.model_max_length
        )
        for field_name in fields_to_stride:
            if max_length_when_striding is None:
                raise ValueError(
                    "Cannot stride if striding length is not provided; "
                    "Please either provide `stride_max_length`, "
                    "`max_{source/target}_length` or "
                    "`tokenizer.model_max_length`."
                )

            strider_mapper = SingleSequenceStriderMapper(
                field_to_stride=field_name,
                max_length=max_length_when_striding,
                stride=stride_step,
            )
            self.chain(strider_mapper)

        if fields_to_truncate:
            # truncation has to be done globally so that all sequence
            # lengths are accounted for
            source_truncation_mapper = TruncateNFieldsMapper(
                fields_to_truncate=fields_to_truncate,
                fields_to_preserve=fields_to_preserve,
                max_length=max_length,
                strategy=strategy,
                tokenizer=tokenizer,
                length_penalty=sum(len(ps) for ps in prompt_mapper.prompt),
            )
            self.chain(source_truncation_mapper)
