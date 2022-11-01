from typing import Any, Dict, Optional, cast

from necessary import Necessary, necessary

from ..base import SingleBaseMapper, TransformElementType
from ..utils import get_name_and_version

with necessary("promptsource", soft=True) as PROMPTSOURCE_AVAILABLE:
    if PROMPTSOURCE_AVAILABLE:
        import yaml
        from promptsource.templates import DatasetTemplates, Template

with necessary("jinja2", soft=True) as JINJA_AVAILABLE:
    if JINJA_AVAILABLE:
        from jinja2 import Environment, meta


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
        return_multiple_targets: bool = False,
        extra_variables: Optional[Dict[str, Any]] = None,
    ):
        self.template = template
        self.truncate = truncate
        self.highlight_vars = highlight_variables
        self.src_fld_name = source_field_name
        self.tgt_fld_name = target_field_name
        self.return_multi_tgt = return_multiple_targets
        self.extra_vars = extra_variables or {}

        # override the id for the template because by default it uses
        # a randomly generated uuid which makes hashing impossible
        setattr(self.template, "id", 0)

        # abstract syntax tree for the jinja template; we will use it
        # to find all fields that are required by the template
        ast = Environment().parse(self.template.jinja)
        input_fields = sorted(
            var_name
            for var_name in meta.find_undeclared_variables(ast)
            if var_name not in self.extra_vars
        )

        output_fields = [self.src_fld_name]
        if "|||" in self.template.jinja:
            output_fields.append(self.tgt_fld_name)

        super().__init__(
            input_fields=input_fields, output_fields=output_fields
        )

    def __getstate__(self) -> dict:
        """We need to serialize the template using yaml so the hash for this
        mapper is consistent across runs."""
        out = super().__getstate__()
        out["__dict__"]["template"] = yaml.dump(self.template)
        return out

    def __setstate__(self, state: dict) -> None:
        """Because we serialized the template as yaml, we need to
        deserialize before we can use it."""
        super().__setstate__(state)
        self.template = yaml.load(
            state["__dict__"]["template"], Loader=yaml.FullLoader
        )

    def transform(self, data: TransformElementType) -> TransformElementType:
        if self.extra_vars:
            # add any extra variables to the data
            data = {**data, **self.extra_vars}

        src, *tgt = self.template.apply(
            data,
            truncate=self.truncate,
            highlight_variables=self.highlight_vars,
        )
        if self.return_multi_tgt:
            return {self.src_fld_name: src, self.tgt_fld_name: tgt}
        elif len(tgt) == 0:
            return {self.src_fld_name: src}
        elif len(tgt) > 1:
            raise ValueError(
                "Multiple targets, but `return_multiple_targets` is False"
            )
        else:
            return {self.src_fld_name: src, self.tgt_fld_name: tgt[0]}


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
        return_multiple_targets: bool = False,
        extra_variables: Optional[Dict[str, Any]] = None,
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
            return_multiple_targets=return_multiple_targets,
            extra_variables=extra_variables,
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
        return_multiple_targets: bool = False,
        extra_variables: Optional[Dict[str, Any]] = None,
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
            return_multiple_targets=return_multiple_targets,
            extra_variables=extra_variables,
        )
