from smashed.base import SingleBaseMapper, TransformElementType


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
        new_line_break: str = "\n",
        new_para_break: str = "\n\n",
    ):
        """
        Args:
            context_field_name (str, optional): Name of the field containing
                the context. Defaults to "context".
            new_line_break (str, optional): String to use to separate lines
                in the context paragraphs. Defaults to "\\n".
            new_para_break (str, optional): String to use to separate
                paragraphs in the context. Defaults to "\\n\\n".
        """

        self.context_field_name = context_field_name
        self.new_line_break = new_line_break
        self.new_para_break = new_para_break
        super().__init__(
            input_fields=[context_field_name],
            output_fields=[context_field_name],
        )

    def transform(self, data: TransformElementType) -> TransformElementType:
        if isinstance(data["context"], str):
            return data
        elif isinstance(data["context"], list):
            data["context"] = self.new_para_break.join(
                self.new_line_break.join(section)
                if isinstance(section, list)
                else section
                for section in data["context"]
            )
            return data
        else:
            raise ValueError(
                "context must be either a string or a list of strings,"
                f' but it is {type(data["context"])}'
            )
