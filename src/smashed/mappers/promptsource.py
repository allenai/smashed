import re
from functools import reduce
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

from necessary import Necessary, necessary

from ..base.mappers import (
    BatchedBaseMapper,
    ChainableMapperMixIn,
    SingleBaseMapper,
    TransformElementType,
)
from ..utils import get_name_and_version

with necessary("promptsource", soft=True) as PROMPTSOURCE_AVAILABLE:
    if PROMPTSOURCE_AVAILABLE:
        import yaml
        from promptsource.templates import DatasetTemplates, Template

with necessary("jinja2", soft=True) as JINJA_AVAILABLE:
    if JINJA_AVAILABLE:
        from jinja2 import Environment, meta


__all__ = [
    "PromptsourceMapper",
    "JinjaMapper",
    "FewShotJinjaMapper",
]

VARSHOTS = "__shots__"


@Necessary(
    "promptsource",
    message="{module_name} missing. Fix with 'pip install smashed[prompting]'",
)
class PromptsourceMixin(ChainableMapperMixIn):
    def __init__(
        self,
        template: "Template",
        output_source_field_name: str = "source",
        output_target_field_name: str = "target",
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
            output_source_field_name (str, optional): the name of the field
                in the returned dictionary of samples that will contain the
                source sequence. Defaults to "source".
            output_target_field_name (str, optional): the name of the field
                in the returned dictionary of samples that will contain the
                target sequence. Defaults to "target".
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

        # override the id for the template because by default it uses
        # a randomly generated uuid which makes hashing impossible
        setattr(self.template, "id", 0)

        self.src_fld_name = output_source_field_name
        self.tgt_fld_name = output_target_field_name
        self.return_multiple_targets = return_multiple_targets
        self.extra_vars = extra_variables or {}

        # merge all fields from source and targets portion of the template
        input_fields: Set[str] = reduce(
            lambda t, s: t.union(s), self.approx_input_fields, set()
        )

        # the output field only contains the target field if the template
        # has a target portion.
        output_fields = [self.src_fld_name]
        if "|||" in self.template.jinja:
            output_fields.append(self.tgt_fld_name)

        super().__init__(
            input_fields=input_fields, output_fields=output_fields
        )

    @staticmethod
    def get_vars_from_txt(text: str) -> Set[str]:
        return meta.find_undeclared_variables(Environment().parse(text))

    @property
    def approx_input_fields(self) -> Tuple[Set[str], ...]:
        """A tuple of sets of input fields that are required by the
        template.

        The first set contains input fields that are
        in the source part of the template (i.e. before the control
        sequence `|||`); subsequent sets contain input fields that
        are in the targets.

        This is a conservative estimate of the input fields required,
        since we can't parse out cases where for loops or if statements
        are used, nor cases where members of a variable are accessed.
        """
        return tuple(
            set(
                field
                for field in self.get_vars_from_txt(t)
                if field not in self.extra_vars
            )
            for t in self.template.jinja.split("|||")
        )

    @property
    def template_text(self) -> Tuple[str, ...]:
        """The text of the template, with all variables and
        control sequences removed."""
        return tuple(
            re.sub(r"\{(%|\{|#).+?(#|%|\})\}", "", t)
            for t in self.template.jinja.split("|||")
        )

    @property
    def has_target(self) -> bool:
        """Whether the template has one or more target sequence."""
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

    def apply_template(self, data: Dict[str, Any]) -> Sequence[str]:
        """Given a dictionary of data, apply the template to generate
        source sequence and target sequence(s)."""

        if self.extra_vars:
            # add any extra variables to the data
            data = {**data, **self.extra_vars}

        return self.template.apply(
            data,
            truncate=self.truncate,
            highlight_variables=self.highlight_vars,
        )

    def format_output(
        self, output: Sequence[str]
    ) -> Dict[str, Union[str, List[str]]]:
        """Given a list of source and target sequences, format the output
        as a dictionary of samples; if `return_multiple_targets` is True,
        then the target field will be a list of strings, otherwise it will
        be a single string."""

        # unpack for convenience; we will have to slice anyway later
        src, *tgt = output

        if self.return_multiple_targets:
            # ok to return multiple targets, so we return a list
            return {self.src_fld_name: src, self.tgt_fld_name: tgt}

        if len(tgt) == 0:
            # no target, so just return the source
            return {self.src_fld_name: src}

        if len(tgt) > 1:
            # we want to return a single target, but there are multiple!
            # therefore, we raise an error.
            raise ValueError(
                "Multiple targets, but `return_multiple_targets` is False"
            )

        return {self.src_fld_name: src, self.tgt_fld_name: tgt[0]}


class SingleTransformPromptsourceMixin(PromptsourceMixin, SingleBaseMapper):
    # We need this class pretty much just so that we can inherit from
    # SingleBaseMapper.
    def transform(self, data: TransformElementType) -> TransformElementType:
        encoded = self.apply_template(data)  # type: ignore
        return self.format_output(encoded)


class PromptsourceMapper(SingleTransformPromptsourceMixin):
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
            output_source_field_name=source_field_name,
            output_target_field_name=target_field_name,
            truncate=truncate,
            highlight_variables=highlight_variables,
            return_multiple_targets=return_multiple_targets,
            extra_variables=extra_variables,
        )


class JinjaMapper(SingleTransformPromptsourceMixin):
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
            name=name,
            reference=(reference or get_name_and_version()),
            metadata=metadata,
        )
        super().__init__(
            template=template,
            output_source_field_name=source_field_name,
            output_target_field_name=target_field_name,
            truncate=truncate,
            highlight_variables=highlight_variables,
            return_multiple_targets=return_multiple_targets,
            extra_variables=extra_variables,
        )


class FewShotJinjaMapper(PromptsourceMixin, BatchedBaseMapper):
    def __init__(
        self,
        jinja: str,
        num_shots: Union[int, Literal["max"]],
        name: Optional[str] = None,
        reference: Optional[str] = None,
        metadata: Optional["Template.Metadata"] = None,
        keep_last: bool = False,
        output_source_field_name: str = "source",
        output_target_field_name: str = "target",
        truncate: bool = False,
        highlight_variables: bool = False,
        return_multiple_targets: bool = False,
        extra_variables: Optional[Dict[str, Any]] = None,
    ):
        """Uses a jinja to generate source and target sequence;
        in the returned dictionary of samples, the source sequence is stored
        under the key `source_field_name` and the target sequence is stored
        under the key `target_field_name`. If the template does not contain
        the control sequence `|||`, then no target sequence is generated.

        Args:
            jinja (str): the jinja template to use. The template can access
                the data in each sample; the name of fields in the datasets
                are available as variables in the template. A special
                variable __shots__ is available, which contains all the shots
                for the sample.
            num_shots (Union[int, Literal['max']]): the number of samples to
                use for each sample. If set to 'max', then all the samples
                in the dataset are used.
            name (Optional[str], optional): the name of the template. Defaults
                to None.
            reference (Optional[str], optional): the reference ID for the
                template. Defaults to None.
            metadata (Optional["Template.Metadata"], optional): the metadata
                for the template. Defaults to None.
            keep_last (bool, optional): whether to keep the last shot in the
                dataset if we have leftover samples less than the number of
                shots. Defaults to False.
            output_source_field_name (str, optional): the name of the field
                in the returned dictionary of samples that will contain the
                source sequence. Defaults to "source".
            output_target_field_name (str, optional): the name of the field
                in the returned dictionary of samples that will contain the
                target sequence. Defaults to "target".
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
        if num_shots != "max" and not (
            isinstance(num_shots, int) and num_shots >= 0
        ):
            raise ValueError(
                "number_of_shots must be a non-negative integer or 'max', "
                f"but got {num_shots}"
            )

        if VARSHOTS not in self.get_vars_from_txt(jinja):
            raise KeyError(
                f"the jinja template must contain the variable {VARSHOTS}"
            )

        template = Template(
            jinja=jinja,
            name=name,
            reference=(reference or get_name_and_version()),
            metadata=metadata,
        )

        # mypy complains if we don't retype num_shots
        self.num_shots: Union[int, Literal["max"]] = num_shots

        # due to how "max" works, we always need to keep the batch
        # when in "max" mode, otherwise we will return an empty dataset
        self.keep_last: bool = keep_last or num_shots == "max"

        super().__init__(
            template=template,
            output_source_field_name=output_source_field_name,
            output_target_field_name=output_target_field_name,
            truncate=truncate,
            highlight_variables=highlight_variables,
            return_multiple_targets=return_multiple_targets,
            extra_variables=extra_variables,
        )

    @property
    def approx_input_fields(self) -> Tuple[Set[str], ...]:
        return tuple(
            set(f for f in fields if f != VARSHOTS)
            for fields in super().approx_input_fields
        )

    def transform(
        self, data: Iterable[TransformElementType]
    ) -> Iterable[TransformElementType]:

        accumulator: List[TransformElementType] = []

        for sample in data:
            if self.num_shots == "max" or len(accumulator) < self.num_shots:
                accumulator.append(sample)
            else:
                output = self.apply_template({**sample, VARSHOTS: accumulator})
                accumulator = []
                yield self.format_output(output)

        if self.keep_last and len(accumulator) > 0:
            # we yield the last bit of the dataset; might have
            # fewer than self.num_shots samples

            # use the last as the non-context sample
            *accumulator, sample = accumulator

            output = self.apply_template({**sample, VARSHOTS: accumulator})
            yield self.format_output(output)
