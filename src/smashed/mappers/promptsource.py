from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, cast

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
        """Uses a promptsource template to generate source and target sequence;
        in the returned dictionary of samples, the source sequence is stored
        under the key `source_field_name` and the target sequence is stored
        under the key `target_field_name`. If the template does not contain
        the control sequence `|||`, then no target sequence is generated.
        Args:
            template (promptsource.templates.Template): the promptsource
                template to use.
            source_field_name (str, optional): the name of the field in the
                returned dictionary of samples that will contain the source
                sequence. Defaults to "source".
            target_field_name (str, optional): the name of the field in the
                returned dictionary of samples that will contain the target
                sequence. Defaults to "target".
            truncate (bool, optional): whether to truncate the source and
                target sequences to the maximum length allowed by
                the promptsource library. Defaults to False.
            highlight_variables (bool, optional): whether to highlight the
                variables in the source and target sequences with special
                html tags. Defaults to False.
            return_multiple_targets (bool, optional): whether to return
                a list of target sequences for each sample. Defaults to False.
                If the template returns multiple targets, but this argument
                is set to False, then only the first target is returned.
            extra_variables (Optional[Dict[str, Any]], optional): a dictionary
                of extra variables that will be passed to the promptsource
                template. Defaults to None.
        """

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

        output_fields = [self.src_fld_name]
        if "|||" in self.template.jinja:
            output_fields.append(self.tgt_fld_name)

        input_src_fields, input_tgt_fields = self.approximate_input_fields
        super().__init__(
            input_fields=set(input_src_fields + input_tgt_fields),
            output_fields=output_fields,
        )

    def _approximate_input_fields(self, jinja_txt: str) -> List[str]:
        ast = Environment().parse(jinja_txt)
        return sorted(
            var_name
            for var_name in meta.find_undeclared_variables(ast)
            if var_name not in self.extra_vars
        )

    @property
    def approximate_input_fields(self) -> Tuple[List[str], List[str]]:
        """Input fields that are likely to be required by the template;
        It is approximate because we ignore nested variables."""

        source_template, *target_templates = self.template.jinja.split("|||")
        source_fields = self._approximate_input_fields(source_template)
        target_fields = sorted(
            set(
                chain.from_iterable(
                    self._approximate_input_fields(template)
                    for template in target_templates
                )
            )
        )
        return source_fields, target_fields

    def _approximate_text_from_template(self, txt: str) -> str:
        return "".join(part.split("}}")[-1] for part in txt.split("{{"))

    @property
    def approximate_prompt_text(self) -> Tuple[str, List[str]]:
        """The prompt without the variables; it is approximate because
        we might not be able to remove all variables."""

        source_template, *target_templates = self.template.jinja.split("|||")

        source_str = self._approximate_text_from_template(source_template)
        target_str = [
            self._approximate_text_from_template(template)
            for template in target_templates
        ]
        return source_str, target_str

    @property
    def has_target(self) -> bool:
        return "|||" in self.template.jinja

    def __getstate__(self) -> dict:
        """We need to serialize thve template using yaml so the hash for this
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
        """Use one of the existing promptsource templates to generate
        source and target sequences for a dataset. See the promptsource
        repository for a list of available templates:
        https://github.com/bigscience-workshop/promptsource

        Args:
            dataset_name (str): the name of the dataset to use.
            template_name (str): the name of the template to use.
            subset_name (Optional[str], optional): the name of the subset
                to use. Defaults to None.
            source_field_name (str, optional): the name of the field in the
                returned dictionary of samples that will contain the source
                sequence. Defaults to "source".
            target_field_name (str, optional): the name of the field in the
                returned dictionary of samples that will contain the target
                sequence. Defaults to "target".
            truncate (bool, optional): whether to truncate the source and
                target sequences to the maximum length allowed by
                the promptsource library. Defaults to False.
            highlight_variables (bool, optional): whether to highlight the
                variables in the source and target sequences with special
                html tags. Defaults to False.
            return_multiple_targets (bool, optional): whether to return
                a list of target sequences for each sample. Defaults to False.
                If the template returns multiple targets, but this argument
                is set to False, then only the first target is returned.
            extra_variables (Optional[Dict[str, Any]], optional): a dictionary
                of extra variables that will be passed to the promptsource
                template. Defaults to None.
        """

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
        """Use a custom jinja template to obtain a template from the
        promptsource library. See the jinja documentation for a list of
        language features and syntax: https://jinja.palletsprojects.com/

        Args:
            jinja (str): the jinja template to use. The template can access
                the data in each sample; the name of fields in the datasets
                are available as variables in the template.
            name (Optional[str], optional): the name of the template. Defaults
                to None.
            reference (Optional[str], optional): the reference for the
                template. Defaults to None.
            metadata (Optional["Template.Metadata"], optional): the metadata
                for the template. Defaults to None.
            source_field_name (str, optional): the name of the field in the
                returned dictionary of samples that will contain the source
                sequence. Defaults to "source".
            target_field_name (str, optional): the name of the field in the
                returned dictionary of samples that will contain the target
                sequence. Defaults to "target".
            truncate (bool, optional): whether to truncate the source and
                target sequences to the maximum length allowed by
                the promptsource library. Defaults to False.
            highlight_variables (bool, optional): whether to highlight the
                variables in the source and target sequences with special
                html tags. Defaults to False.
            return_multiple_targets (bool, optional): whether to return
                a list of target sequences for each sample. Defaults to False.
                If the template returns multiple targets, but this argument
                is set to False, then only the first target is returned.
            extra_variables (Optional[Dict[str, Any]], optional): a dictionary
                of extra variables that will be passed to the promptsource
                template. Defaults to None.
        """
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
