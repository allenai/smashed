import inspect
from typing import Any, List, Union

from ftfy import TextFixerConfig, fix_text

from ..base import SingleBaseMapper, TransformElementType


class FtfyMapper(SingleBaseMapper):
    """Uses ftfy to fix text encoding and general weirdness issues."""

    def __init__(
        self,
        input_fields: Union[str, List[str]],
        **ftfy_kwargs: Any,
    ) -> None:
        """Initialize the mapper.

        Args:
            input_fields (Union[str, List[str]]): The fields to fix;
                if a string, it is assumed to be a single field.
            ftfy_kwargs (Any): Any keyword arguments to use to create a
                ftfy.TextFixerConfig object. See the ftfy documentation
                for more information.
        """

        if isinstance(input_fields, str):
            input_fields = [input_fields]

        self.fields_to_fix = set(input_fields)

        # check if options for ftfy are valid
        valid_ftfy_args = set(inspect.getfullargspec(TextFixerConfig).args)
        for arg_name in ftfy_kwargs:
            if arg_name not in valid_ftfy_args:
                raise ValueError(
                    f"Invalid argument for ftfy.TextFixerConfig: {arg_name}"
                )
        self.ftfy_config = TextFixerConfig(**ftfy_kwargs)

        super().__init__(input_fields=input_fields, output_fields=input_fields)

    def transform(self, data: TransformElementType) -> TransformElementType:
        return {
            field: (
                fix_text(value, config=self.ftfy_config)
                if field in self.fields_to_fix
                else value
            )
            for field, value in data.items()
        }
