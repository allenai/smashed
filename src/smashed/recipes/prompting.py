from typing import Dict, Literal, Optional, Sequence, TypeVar, Union

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..base.mappers import ChainableMapperMixIn
from ..base.recipes import BaseRecipe
from ..mappers.fields import RenameFieldsMapper
from ..mappers.prompting import (
    EncodeFieldsMapper,
    FillEncodedPromptMapper,
    TruncateMultipleFieldsMapper,
)
from ..mappers.shape import SingleSequenceStriderMapper

C = TypeVar("C", bound=ChainableMapperMixIn)


class PromptingRecipe(BaseRecipe):

    # these make for easy replacement of the mappers in subclasses
    # of this recipe

    def encoder_mapper(self, **kwargs) -> EncodeFieldsMapper:
        return EncodeFieldsMapper(**kwargs)

    def prompt_mapper(self, **kwargs) -> FillEncodedPromptMapper:
        return FillEncodedPromptMapper(**kwargs)

    def truncate_mapper(self, **kwargs) -> TruncateMultipleFieldsMapper:
        return TruncateMultipleFieldsMapper(**kwargs)

    def strider_mapper(self, **kwargs) -> SingleSequenceStriderMapper:
        return SingleSequenceStriderMapper(**kwargs)

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        source_template: str,
        source_add_bos_token: bool = True,
        source_add_eos_token: bool = False,
        target_template: Optional[str] = None,
        target_add_bos_token: bool = False,
        target_add_eos_token: bool = False,
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
        extra_keep_field_names: Union[
            None, Sequence[str], Dict[str, str]
        ] = None,
        extra_encode_fields: Optional[Sequence[str]] = None,
    ):
        """A recipe that creates chained mappers for prompting tasks.

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to use.
            source_template (str): The template for the source prompt; see
                FillEncodedPromptMapper for more details.
            source_add_bos_token (bool, optional): Whether to add the BOS token
                to the source prompt. Defaults to True.
            source_add_eos_token (bool, optional): Whether to add the EOS token
                to the source prompt. Defaults to False.
            target_template (Optional[str], optional): The template for the
                target prompt; see FillEncodedPromptMapper for more details.
                If not provided, we only use templates for the source.
                Defaults to None.
            target_add_bos_token (bool, optional): Whether to add the BOS token
                to the target prompt. Defaults to False.
            target_add_eos_token (bool, optional): Whether to add the EOS token
                to the target prompt. Defaults to False.
            fields_to_truncate (Optional[Sequence[str]], optional): The fields
                to truncate. If not provided, we will not truncate any fields.
                Defaults to None.
            fields_to_stride (Optional[Sequence[str]], optional): The fields
                to stride. If not provided, we will not stride any fields.
                Defaults to None.
            stride_max_length (Optional[int], optional): The maximum length
                of the stride. If not provided, we will use the maximum length
                of the tokenizer. Defaults to None.
            stride_step (Optional[int], optional): The step size of the stride.
                If not provided, we will use the stride max length. Defaults to
                None.
            target_output_name ("labels" or "decoder_input_ids", optional): The
                name of field where to save the output of the target prompt.
                Defaults to "labels".
            is_split_into_words (bool, optional): Whether the input is split
                into words. Defaults to False.
            max_source_length (Optional[int], optional): The maximum length
                of the source prompt. If not provided, we will use the maximum
                length of the tokenizer. Defaults to None.
            max_target_length (Optional[int], optional): The maximum length
                of the target prompt. If not provided, we will use the maximum
                length of the tokenizer. Defaults to None.
            strategy ("longest" or "uniform", optional): The strategy
                to use for truncation. Defaults to "longest".
            return_attention_mask (bool, optional): Whether to return the
                attention mask. Defaults to True.
            return_token_type_ids (bool, optional): Whether to return the
                token type ids. Defaults to False.
            extra_keep_field_names (List[str] or  Dict[str, str], optional):
                The extra fields to keep. If a list is provided, we will keep
                the fields with the same name. If a dictionary is provided,
                we will keep the fields with the name specified in the
                dictionary. Defaults to None.
            extra_encode_fields (Optional[Sequence[str]], optional): The extra
                fields to encode. If not provided, we will not encode any extra
                fields besides the ones indicated in the source or target
                template. Defaults to None.
        """
        super().__init__()

        fields_to_truncate = fields_to_truncate or []
        fields_to_stride = fields_to_stride or []

        extra_keep_field_names = extra_keep_field_names or []
        if not isinstance(extra_keep_field_names, dict):
            extra_keep_field_names = {f: f for f in extra_keep_field_names}

        extra_encode_fields = extra_encode_fields or []

        source_prompt_mapper = self.prompt_mapper(
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
            target_prompt_mapper = self.prompt_mapper(
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
        pipeline: ChainableMapperMixIn = self.encoder_mapper(
            fields_to_encode=fields_to_encode,
            tokenizer=tokenizer,
            is_split_into_words=is_split_into_words,
        )

        # for the source, this adds truncation / striding depending on
        # whether they are needed or not.
        pipeline = self._add_truncation_and_striding(
            pipeline=pipeline,
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
            pipeline = self._add_truncation_and_striding(
                pipeline=pipeline,
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
        pipeline.chain(source_prompt_mapper)

        # we need to rename some fields to make sure there is no conflict
        # between the source and target prompts; further this will eliminate
        # all fields that are not needed for this recipe.
        rename_fields_map = {
            **{k: k for k in source_prompt_mapper.output_fields},
            **{k: v for k, v in extra_keep_field_names.items()},
        }

        if target_prompt_mapper:
            # we add the target prompt template filler to the chain in case
            # we have a target template
            pipeline.chain(target_prompt_mapper)

            # some models want the target sequence to be called "labels",
            # others want it to be called "decoder_input_ids"; we take care
            # of this here.
            if target_output_name == "labels":
                rename_fields_map["decoder_input_ids"] = "labels"
            else:
                rename_fields_map["decoder_input_ids"] = "decoder_input_ids"

        # finally, this renames the fields to handle aforementioned conflicts
        pipeline.chain(
            RenameFieldsMapper(
                rename_fields_map=rename_fields_map, remove_rest=True
            )
        )

        self.chain(pipeline)

    def _add_truncation_and_striding(
        self,
        pipeline: C,
        prompt_mapper: FillEncodedPromptMapper,
        tokenizer: PreTrainedTokenizerBase,
        all_fields_to_truncate: Sequence[str],
        all_fields_to_stride: Sequence[str],
        strategy: Union[Literal["longest"], Literal["uniform"]],
        max_length: Optional[int] = None,
        stride_max_length: Optional[int] = None,
        stride_step: Optional[int] = None,
    ) -> C:

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

            strider_mapper = self.strider_mapper(
                field_to_stride=field_name,
                max_length=max_length_when_striding,
                stride=stride_step,
            )
            pipeline.chain(strider_mapper)

        if fields_to_truncate:
            # truncation has to be done globally so that all sequence
            # lengths are accounted for
            trunc_mapper = self.truncate_mapper(
                fields_to_truncate=fields_to_truncate,
                fields_to_preserve=fields_to_preserve,
                max_length=max_length,
                strategy=strategy,
                tokenizer=tokenizer,
                length_penalty=sum(len(ps) for ps in prompt_mapper.prompt),
            )
            pipeline.chain(trunc_mapper)

        return pipeline
