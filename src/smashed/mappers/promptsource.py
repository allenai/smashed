from typing import Optional, cast
from ..base import SingleBaseMapper, TransformElementType
from ..utils import get_name_and_version

from necessary import necessary, Necessary

if necessary("promptsource", soft=True):
    from promptsource.templates import Template
    from promptsource.templates import DatasetTemplates
    from jinja2 import Environment, meta

PS_MISSING_MSG = (
    "You must install promptsource to use this mapper. "
    "You can do so by running `pip install promptsource` or "
    "`pip install smashed[datasets]`."
)


@Necessary("promptsource", message=PS_MISSING_MSG)
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
            highlight_variables=self.highlight_variables
        )
        return {"source": source, "target": target}


@Necessary("promptsource", message=PS_MISSING_MSG)
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
            dataset_name=dataset_name, subset_name=subset_name,
        )[template_name]

        super().__init__(
            template=template,
            source_field_name=source_field_name,
            target_field_name=target_field_name,
            truncate=truncate,
            highlight_variables=highlight_variables,
        )


@Necessary("promptsource", message=PS_MISSING_MSG)
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
