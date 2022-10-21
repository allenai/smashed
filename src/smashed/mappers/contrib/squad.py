from typing import Optional

from ...base import SingleBaseMapper, TransformElementType


__all__ = [
    "AddEvidencesLocation",
    "ConcatenateContextMapper",
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
    """A mapper that removes duplicate answers from the answer field """

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


class AddEvidencesLocation(SingleBaseMapper):
    """A mapper that adds the location of  """
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

    def transform(
        self, data: TransformElementType
    ) -> TransformElementType:
        return {
            "locations": [
                data[self.context_field].find(evidence)
                for evidence in data[self.evidence_field]
            ]
        }
