from smashed.base import SingleBaseMapper, TransformElementType


class ConcatenateContextMapper(SingleBaseMapper):
    def __init__(self, context_field_name: str = "context"):
        self.context_field_name = context_field_name
        super().__init__(
            input_fields=[context_field_name],
            output_fields=[context_field_name],
        )

    def transform(self, data: TransformElementType) -> TransformElementType:
        if isinstance(data["context"], str):
            return data
        elif isinstance(data["context"], list):
            data["context"] = "\n\n".join(
                "\n".join(section) if isinstance(section, list) else section
                for section in data["context"]
            )
            return data
        else:
            raise ValueError(
                "context must be either a string or a list of strings,"
                f' but it is {type(data["context"])}'
            )
