from bisect import bisect_left, bisect_right
from typing import Any, Literal, Optional, Sequence, Tuple, TypeVar, Union

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from smashed.base import BaseRecipe, SingleBaseMapper, TransformElementType
from smashed.base.mappers import ChainableMapperMixIn
from smashed.mappers import (
    ChangeFieldsMapper,
    EncodeFieldsMapper,
    FillEncodedPromptMapper,
    MakeFieldMapper,
    RangeToMaskMapper,
    SingleSequenceStriderMapper,
    UnpackingMapper,
)
from smashed.recipes.prompting import PromptingRecipe

__all__ = [
    "AddEvidencesLocationMapper",
    "ConcatenateContextMapper",
    "EncoderWithEvidenceLocationMapper",
    "ReplaceIfNoEvidencePromptMapper",
    "SquadPromptTrainRecipe",
    "SquadPromptValidRecipe",
    "StriderWithEvidenceLocationMapper",
    "UniqueAnswerMapper",
]


class ConcatenateContextMapper(SingleBaseMapper):
    """Concatenates the various fields in the context into a single string

    Context representation can be one of the following:
        1. a string
        2. a list of strings
        3. a list of lists of strings

    The first one is handled by simply returning the data; the second one is
    handled by joining the strings with a `new_line_break` separator; the third
    one is handled by joining the strings with a `new_line_break` separator.
    """

    def __init__(
        self,
        context_field_name: str = "context",
        section_bos: str = "",
        section_eos: str = "\n",
        paragraph_bos: str = "\n\n",
        paragraph_eos: str = "\n",
        header_bos: Optional[str] = None,
        header_eos: Optional[str] = None,
        document_bos: str = "",
        document_eos: str = "",
    ):
        """
        Args:
            context_field_name (str): the name of the field containing the
                context. Defaults to "context".
            section_bos (str): the beginning of section token. Defaults to ""
                (empty string).
            section_eos (str): the end of section token. Defaults to "\\n"
                (a single new line).
            paragraph_bos (str): the beginning of paragraph token.
                Defaults to "\\n\\n" (two new lines).
            paragraph_eos (str): the end of paragraph token. Defaults to "\\n"
                (a single new line).
            header_bos (str, optional): the beginning of header token; if None,
                the same as section_bos. Defaults to None.
            header_eos (str, optional): the end of header token; if None,
                the same as section_eos. Defaults to None.
            document_bos (str): the beginning of document token. Defaults to
                '' (empty string).
            document_eos (str): the end of document token. Defaults to ''
                (empty string).
        """
        self.ctx_fld = context_field_name
        self.sec_bos = section_bos
        self.sec_eos = section_eos
        self.par_bos = paragraph_bos
        self.par_eos = paragraph_eos
        self.hdr_bos = header_bos or self.par_bos
        self.hdr_eos = header_eos or self.par_eos
        self.doc_bos = document_bos
        self.doc_eos = document_eos

        super().__init__(
            input_fields=[context_field_name],
            output_fields=[context_field_name],
        )

    def transform(self, data: TransformElementType) -> TransformElementType:
        if isinstance(data[self.ctx_fld], str):
            return data
        elif isinstance(data[self.ctx_fld], list):
            sections = []
            for sec in data[self.ctx_fld]:
                if sec is None:
                    continue

                elif isinstance(sec, str):
                    sections.append(self.sec_bos + sec + self.sec_eos)

                elif isinstance(sec, list):
                    for i, para in enumerate(sec):
                        if para is None:
                            continue
                        if i == 0:
                            sections.append(self.hdr_bos + para + self.hdr_eos)
                        else:
                            sections.append(self.par_bos + para + self.par_eos)
                else:
                    raise ValueError(f"Invalid type for section: {type(sec)}")

            data[self.ctx_fld] = (
                self.doc_bos + "".join(sections) + self.doc_eos
            )
            return data
        else:
            raise ValueError(
                "context must be either a string or a list of strings,"
                f" but it is {type(data[self.ctx_fld])}"
            )


class UniqueAnswerMapper(SingleBaseMapper):
    """A mapper that removes duplicate answers from the answer field"""

    answer_field: str

    def __init__(self, answer_field: str = "answers"):
        super().__init__()
        self.answer_field = answer_field

    def transform(self, data: TransformElementType) -> TransformElementType:
        data[self.answer_field] = [
            # we use fromkeys to remove duplicates because it
            # preserves the order of the list
            t
            for t in dict.fromkeys(data[self.answer_field])
        ]
        return data


class AddEvidencesLocationMapper(SingleBaseMapper):
    """A mapper that adds the location of"""

    def __init__(
        self,
        context_field: str = "context",
        evidence_field: str = "evidences",
        location_field: str = "locations",
    ) -> None:
        self.context_field = context_field
        self.evidence_field = evidence_field
        self.location_field = location_field
        super().__init__(
            input_fields=[evidence_field],
            output_fields=[location_field],
        )

    def transform(self, data: TransformElementType) -> TransformElementType:
        out = {
            "locations": [
                (
                    (ev := data[self.context_field].find(evidence)),
                    ev + len(evidence) if ev >= 0 else -1,
                )
                for grouped in data[self.evidence_field]
                for evidence in (
                    grouped if isinstance(grouped, list) else (grouped,)
                )
            ]
        }
        # if len(out["locations"]) == 0:
        #     # this is necessary otherwise huggingface
        #     out["locations"] = [[-1, -1]]

        return out


class EncoderWithEvidenceLocationMapper(EncodeFieldsMapper):
    """Regular encoder but shifts the locations in the locations field
    based on the encoding of the context field"""

    def __init__(
        self,
        *args,
        context_field: str = "context",
        location_field: str = "locations",
        fields_to_encode: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        kwargs["fields_to_return_offset_mapping"] = [context_field]
        kwargs["fields_to_encode"] = [context_field] + list(
            fields_to_encode or []
        )
        super().__init__(*args, **kwargs)

        self.context_field = context_field
        self.location_field = location_field

        # add location field to expected input fields
        if location_field not in self.input_fields:
            self.input_fields = (location_field, *self.input_fields)

        # remove the field with the context_offsets because we are going
        # to pop it out!
        self.output_fields = tuple(
            f
            for f in self.output_fields
            if f != f"{self.offset_prefix}_{self.context_field}"
        )

        self.chain(
            RangeToMaskMapper(
                mask_field_name=self.location_field,
                reference_field_name=self.context_field,
                locations_field_name=self.location_field,
            )
        )
        # self.

    def transform(self, data: TransformElementType) -> TransformElementType:
        out = super().transform(data)
        offsets = out.pop(f"{self.offset_prefix}_{self.context_field}")
        start_offsets, end_offsets = zip(*offsets)

        # this is where we will add the shifted locations
        out[self.location_field] = []

        for start, end in data[self.location_field]:
            if start > 0:
                pos = bisect_right(start_offsets, start)
                start, _ = offsets[pos - 1]
            else:
                start = -1
            if end > 0:
                pos = bisect_left(end_offsets, end)
                _, end = offsets[pos]
            else:
                end = -1

            out[self.location_field].append([start, end])

        return out


class StriderWithEvidenceLocationMapper(SingleSequenceStriderMapper):
    def __init__(
        self,
        *args,
        context_field: str = "context",
        location_field: str = "locations",
        field_to_stride: Optional[Union[str, Sequence[str]]] = None,
        **kwargs,
    ):
        field_to_stride = (
            [field_to_stride]
            if isinstance(field_to_stride, str)
            else (field_to_stride or [])
        )
        unique_field_to_stride = set(
            (context_field, location_field, *field_to_stride)
        )
        kwargs["field_to_stride"] = sorted(unique_field_to_stride)
        super().__init__(*args, **kwargs)


class ReplaceIfNoEvidencePromptMapper(FillEncodedPromptMapper):
    def __init__(
        self,
        *args,
        location_field: str = "locations",
        target_field: str = "labels",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.location_field = location_field
        self.target_field = target_field

    def transform(self, data: TransformElementType) -> TransformElementType:
        if sum(data[self.location_field]) > 0:
            encoded_target = data[self.target_field]
        else:
            encoded_target = sum(
                (ps.fill_encoded(data) for ps in self.prompt), []
            )

            try:
                if len(data[self.target_field]) and isinstance(
                    data[self.target_field][0], list
                ):
                    # at test time, we have multiple answers!
                    encoded_target = [encoded_target]
            except KeyError:
                breakpoint()

        return {self.target_field: encoded_target}


class _SquadPromptingRecipe(PromptingRecipe):
    def encoder_mapper(self, **kwargs) -> EncodeFieldsMapper:
        return EncoderWithEvidenceLocationMapper(
            context_field=self.context_field,
            location_field=self.location_field,
            **kwargs,
        )

    def strider_mapper(self, **kwargs) -> SingleSequenceStriderMapper:
        return StriderWithEvidenceLocationMapper(
            context_field=self.context_field,
            location_field=self.location_field,
            **kwargs,
        )

    def __init__(
        self,
        *args,
        tokenizer: PreTrainedTokenizerBase,
        context_field: str = "context",
        location_field: str = "locations",
        **kwargs,
    ):
        self.location_field = location_field
        self.context_field = context_field
        kwargs = {**kwargs, "tokenizer": tokenizer}
        super().__init__(*args, **kwargs)


C = TypeVar("C", bound=ChainableMapperMixIn)


class SquadPromptTrainRecipe(BaseRecipe):
    def unpacking(self, pipeline: C, **kwargs: Any) -> C:
        return pipeline >> UnpackingMapper(**kwargs)

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        source_template: str,
        context_length: int,
        context_stride: int,
        target_template: Optional[str] = None,
        alternative_template: Optional[str] = None,
        target_output_name: Union[
            Literal["labels"], Literal["decoder_input_ids"]
        ] = "labels",
        answer_field: str = "answers",
        context_field: str = "context",
        location_field: str = "locations",
        evidence_field: Union[str, None] = "evidences",
        extras_field: Union[str, None] = "extras",
        id_field: Union[str, None] = "id",
        section_bos: str = "",
        section_eos: str = "\n",
        paragraph_bos: str = "\n\n",
        paragraph_eos: str = "\n",
        header_bos: Optional[str] = None,
        header_eos: Optional[str] = None,
        document_bos: str = "",
        document_eos: str = "",
        source_add_bos_token: bool = False,
        source_add_eos_token: bool = False,
        target_add_bos_token: bool = False,
        target_add_eos_token: bool = False,
        extra_keep_field_names: Optional[Sequence[str]] = None,
        extra_encode_fields: Optional[Sequence[str]] = None,
    ):
        super().__init__()

        fields_to_unpack: Tuple[str, ...] = (
            answer_field,
            *((evidence_field,) if evidence_field else []),
            *((extras_field,) if extras_field else []),
        )

        pipeline = ConcatenateContextMapper(
            context_field_name=context_field,
            section_bos=section_bos,
            section_eos=section_eos,
            paragraph_bos=paragraph_bos,
            paragraph_eos=paragraph_eos,
            header_bos=header_bos,
            header_eos=header_eos,
            document_bos=document_bos,
            document_eos=document_eos,
        )

        pipeline = self.unpacking(
            pipeline,
            fields_to_unpack=fields_to_unpack,
            ignored_behavior="repeat",
        )

        if evidence_field:
            pipeline = pipeline >> AddEvidencesLocationMapper(
                context_field=context_field,
                evidence_field=evidence_field,
                location_field=location_field,
            )

        pipeline = pipeline >> _SquadPromptingRecipe(
            tokenizer=tokenizer,
            context_field=context_field,
            location_field=location_field,
            source_template=source_template,
            target_template=target_template,
            fields_to_truncate=[context_field],
            fields_to_stride=[context_field],
            stride_max_length=context_length,
            stride_step=context_stride,
            target_output_name=target_output_name,
            source_add_bos_token=source_add_bos_token,
            source_add_eos_token=source_add_eos_token,
            target_add_bos_token=target_add_bos_token,
            target_add_eos_token=target_add_eos_token,
            extra_keep_field_names=(
                *([location_field] if evidence_field else []),
                *([id_field] if id_field else []),
                *(extra_keep_field_names or []),
            ),
            extra_encode_fields=extra_encode_fields,
        )

        if alternative_template:
            pipeline = pipeline >> ReplaceIfNoEvidencePromptMapper(
                target_field=target_output_name,
                location_field=location_field,
                template=alternative_template,
                tokenizer=tokenizer,
            )

        if evidence_field:
            pipeline = pipeline >> ChangeFieldsMapper(
                drop_fields=[location_field]
            )

        self.chain(pipeline)


class SquadPromptValidRecipe(SquadPromptTrainRecipe):
    def __init__(
        self,
        *args,
        tokenizer: PreTrainedTokenizerBase,
        target_output_name: Optional[str] = None,
        answer_field: str = "answers",
        **kwargs,
    ):
        kwargs = {
            **kwargs,
            "tokenizer": tokenizer,
            "target_template": None,
            "target_output_name": answer_field,
            "answer_field": answer_field,
            "extra_keep_field_names": [answer_field],
            "extra_encode_fields": [answer_field],
        }
        super().__init__(*args, **kwargs)

        self.chain(
            MakeFieldMapper(
                field_name="decoder_input_ids",
                value=[tokenizer.pad_token_id],
            )
        )

    def unpacking(self, pipeline: C, **kwargs: Any) -> C:
        # we don't unpack the answers in the validation set
        return pipeline
