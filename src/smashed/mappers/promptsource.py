from copy import deepcopy
from functools import partial
from typing import Dict, Literal, Optional, cast

from necessary import Necessary, necessary

from ..base import SingleBaseMapper, TransformElementType
from ..utils import Nested, get_name_and_version
from ..utils.wordsplitter import (
    BaseWordSplitter,
    BlingFireSplitter,
    WhitespaceSplitter,
)

with necessary("promptsource", soft=True) as PROMPTSOURCE_AVAILABLE:
    if PROMPTSOURCE_AVAILABLE:
        from promptsource.templates import DatasetTemplates, Template

with necessary("jinja2", soft=True) as JINJA_AVAILABLE:
    if JINJA_AVAILABLE:
        from jinja2 import Environment, meta


class TextTruncateMapper(SingleBaseMapper):
    def __init__(self, fields_truncate_map: Dict[str, int]):
        self.fields_to_truncate = tuple(
            (
                Nested.from_str(field),
                partial(self._truncate, truncate_to=truncate_to),
            )
            for field, truncate_to in fields_truncate_map.items()
        )
        # we only check for the first in case of nested fields
        io_fields = [str(spec.key[0]) for spec, _ in self.fields_to_truncate]
        super().__init__(input_fields=io_fields, output_fields=io_fields)

    def _truncate(self, data: str, truncate_to: int) -> str:
        return data[:truncate_to]

    def transform(self, data: TransformElementType) -> TransformElementType:
        data = deepcopy(data)
        for field, truncate_fn in self.fields_to_truncate:
            field.edit(data, truncate_fn)  # type: ignore
        return data


class WordsTruncateMapper(TextTruncateMapper):
    splitter: BaseWordSplitter

    def __init__(
        self,
        fields_truncate_map: Dict[str, int],
        splitter: Literal["blingfire", "whitespace"] = "blingfire",
    ):
        super().__init__(fields_truncate_map)
        if splitter == "blingfire":
            self.splitter = BlingFireSplitter()
        elif splitter == "whitespace":
            self.splitter = WhitespaceSplitter()
        else:
            raise ValueError(f"Unknown splitter: {splitter}")

    def _truncate(self, data: str, truncate_to: int) -> str:
        words = self.splitter(data)
        return " ".join(words[:truncate_to])


@Necessary(
    "promptsource",
    message="{module_name} missing. Fix with 'pip install smashed[prompting]'",
)
class PromptsourceMapper(SingleBaseMapper):
    def __init__(
        self,
        template: "Template",
        source_field_name: str = "source",
        target_field_name: str = "target",
        truncate: bool = False,
        highlight_variables: bool = False,
    ):
        self.template = template
        self.truncate = truncate
        self.highlight_variables = highlight_variables
        self.source_field_name = source_field_name
        self.target_field_name = target_field_name

        # abstract syntax tree for the jinja template; we will use it
        # to find all fields that are required by the template
        ast = Environment().parse(self.template.jinja)

        super().__init__(
            input_fields=sorted(meta.find_undeclared_variables(ast)),
            output_fields=(source_field_name, target_field_name),
        )

    def transform(self, data: TransformElementType) -> TransformElementType:
        source, target = self.template.apply(
            data,
            truncate=self.truncate,
            highlight_variables=self.highlight_variables,
        )
        return {"source": source, "target": target}


@Necessary(
    "promptsource",
    message="{module_name} missing. Fix with 'pip install smashed[prompting]'",
)
class DatasetPromptsourceMapper(PromptsourceMapper):
    def __init__(
        self,
        dataset_name: str,
        template_name: str,
        subset_name: Optional[str] = None,
        source_field_name: str = "source",
        target_field_name: str = "target",
        truncate: bool = False,
        highlight_variables: bool = False,
    ):
        # DatasetTemplates is not well annotated, so though subset_name
        # is optional, it is annotated as `str`, so we need to cast it.
        subset_name = cast(str, subset_name)

        template = DatasetTemplates(
            dataset_name=dataset_name,
            subset_name=subset_name,
        )[template_name]

        super().__init__(
            template=template,
            source_field_name=source_field_name,
            target_field_name=target_field_name,
            truncate=truncate,
            highlight_variables=highlight_variables,
        )


@Necessary(
    "promptsource",
    message="{module_name} missing. Fix with 'pip install smashed[prompting]'",
)
class JinjaPromptsourceMapper(PromptsourceMapper):
    def __init__(
        self,
        jinja: str,
        name: Optional[str] = None,
        reference: Optional[str] = None,
        metadata: Optional["Template.Metadata"] = None,
        source_field_name: str = "source",
        target_field_name: str = "target",
        truncate: bool = False,
        highlight_variables: bool = False,
    ):
        template = Template(
            jinja=jinja,
            name=(name or self.name),
            reference=(reference or get_name_and_version()),
            metadata=metadata,
        )
        super().__init__(
            template=template,
            source_field_name=source_field_name,
            target_field_name=target_field_name,
            truncate=truncate,
            highlight_variables=highlight_variables,
        )
