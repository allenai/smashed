from functools import reduce
from typing import Literal, Optional, Sequence, Set, Union, cast

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..base.recipes import BaseRecipe
from ..mappers.fields import ChangeFieldsMapper
from ..mappers.prompting import TruncateMultipleFieldsMapper
from ..mappers.promptsource import VARSHOTS, FewShotJinjaMapper, JinjaMapper
from ..mappers.text import TextToWordsMapper, WordsToTextMapper
from ..mappers.tokenize import TokenizerMapper


class JinjaRecipe(BaseRecipe):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        jinja_template: str,
        num_shots: Union[int, Literal["max"]] = 0,
        max_source_content_length: Optional[int] = None,
        max_target_content_length: Optional[int] = None,
        truncation_strategy: Literal["longest", "uniform"] = "longest",
        use_words: bool = True,
        source_fields: Optional[Sequence[str]] = None,
        target_fields: Optional[Sequence[str]] = None,
        additional_fields_to_keep: Optional[Sequence[str]] = None,
    ) -> None:
        """A recipe for a pipeline that uses promptsource to format data
        as source/target pairs for model prompting.

        Args:
            tokenizer (PreTrainedTokenizerBase): A tokenizer to use for
                tokenizing the source and target.
            jinja_template (str): A jinja template to use for formatting
                the source and target; we use promptsource to parse the
                template and extract the source and target fields; please
                see the promptsource documentation for more details.
            max_source_content_length (Optional[int], optional): the maximum
                length of the source content (i.e., the content that is given
                as input to the model). If not provided, no truncation will
                be performed. Defaults to None.
            max_target_content_length (Optional[int], optional): the maximum
                length of the target content (i.e., the content that is
                expected as output from the model). If not provided, no
                truncation will be performed. Defaults to None.
            truncation_strategy ("longest" or "uniform"], optional): how to
                perform truncation if the source or target content is longer
                than the maximum length. If "longest", the longest fields
                specified in the template will be truncated first. If
                "uniform", the fields will be truncated uniformly. Defaults
                to "longest".
            use_words (bool, optional): When truncating, whether to use count
                of words or count of characters. Defaults to True, which means
                that we use count of words.
            source_fields (Optional[Sequence[str]], optional): The fields in
                the template that are the source. If not provided, we will
                try to infer them from the template. Defaults to None.
            target_fields (Optional[Sequence[str]], optional): The fields in
                the template that are the target. If not provided, we will
                try to infer them from the template. Defaults to None.
            additional_fields_to_keep (Optional[Sequence[str]], optional):
                After the recipe has been applied, we drop all columns that
                are not 'input_ids', 'attention_mask', or 'labels'. If you
                want to keep additional columns, you can specify them here.
                Defaults to None.
        """

        # must run init before we start using `chain` function to add mappers
        super().__init__()

        # we instantiate the template mapper early on so we can get the text
        # in the prompt that is not variable placeholders; however, we will
        # wait till truncation mappers are added to the pipeline before
        # instantiating the template mapper.

        # The mapper could be a FewShotJinjaMapper or a JinjaMapper, depending
        # on whether it contains the variable `__shots__`.
        template_mapper: Union[JinjaMapper, FewShotJinjaMapper]
        if VARSHOTS in JinjaMapper.get_vars_from_txt(jinja_template):
            template_mapper = FewShotJinjaMapper(
                jinja=jinja_template, num_shots=num_shots
            )
        else:
            template_mapper = JinjaMapper(jinja=jinja_template)

        # if not provided, we try to infer the source and target fields
        source_fields = list(
            source_fields or template_mapper.approx_input_fields[0]
        )
        target_fields = list(
            target_fields
            or reduce(
                lambda t, s: t.union(s),
                template_mapper.approx_input_fields[1:],
                cast(Set[str], set()),  # cast necessary for mypy
            )
        )

        # we get the text used in the prompt for source and target
        # that is not a variable placeholder or control sequence .
        source_text, *target_text = template_mapper.template_text

        if use_words:
            # if we we need to first set up a text -> words splitter for
            # the fields in the template
            text_to_words = TextToWordsMapper(
                fields=source_fields + target_fields
            )
            self.chain(text_to_words)

            # we also need to calculate the lengths in words of the part of
            # the prompt that is not content; that way we can subtract it
            # from the max content length, for both source and target.
            length_src_prompt = len(text_to_words.splitter(source_text))

            # for target, we actually take the max in case there are multiple
            # prompt versions.
            length_tgt_prompt = max(
                [len(text_to_words.splitter(t)) for t in target_text]
                # in case tgt_text is empty, we use 0 as a default value
                or [0]
            )
        else:
            # if we don't use words, we just use the length of the prompt
            # in characters.
            length_src_prompt = len(source_text)
            length_tgt_prompt = len(target_text)

        if max_source_content_length is not None:
            # in case a max length for the source is provided, we need to
            # truncate; first, we decrease the max length by the length of
            # prompt text.
            max_source_content_length -= length_src_prompt

            # we raise if the max length is less than one after accounting
            # for the length of the prompt text.
            if max_source_content_length < 1:
                raise ValueError(
                    f"max_source_content_length must be at least equal to "
                    f"the length of the source prompt ({length_src_prompt})!"
                )

            # finally we add a mapper that truncates the source fields.
            self.chain(
                TruncateMultipleFieldsMapper(
                    fields_to_truncate=source_fields,
                    max_length=max_source_content_length,
                    strategy=truncation_strategy,
                )
            )

        if len(target_text) > 0 and max_target_content_length:
            # we operate here in the same way as for the source, but we
            # only do it if there is a target prompt.
            max_target_content_length -= length_tgt_prompt

            if max_target_content_length < 1:
                raise ValueError(
                    f"max_target_content_length must be at least equal to "
                    f"the length of the target prompt ({length_tgt_prompt})!"
                )

            self.chain(
                TruncateMultipleFieldsMapper(
                    fields_to_truncate=target_fields,
                    max_length=max_target_content_length,
                    strategy=truncation_strategy,
                )
            )

        if use_words:
            # if we used words, we need to convert the fields back to text
            # before filling the template.
            self.chain(WordsToTextMapper(fields=source_fields + target_fields))

        # we only add the template here because we first need to truncate
        # the fields!
        self.chain(template_mapper)

        # tokenize source
        self.chain(
            TokenizerMapper(
                tokenizer=tokenizer,
                input_field="source",
                add_special_tokens=False,
                return_attention_mask=True,
                truncation=True,
            )
        )
        # we need to keep the input_ids and attention_mask fields
        # after the recipe has been applied.
        keep_fields = ["input_ids", "attention_mask"]

        if template_mapper.has_target:
            # tokenize target
            self.chain(
                TokenizerMapper(
                    tokenizer=tokenizer,
                    input_field="target",
                    output_rename_map={"input_ids": "labels"},
                    add_special_tokens=False,
                    return_attention_mask=False,
                    truncation=True,
                )
            )
            # the target is in the labels field, so we need to keep it.
            keep_fields.append("labels")

        if additional_fields_to_keep:
            # this is in case the user wants to keep additional fields
            keep_fields.extend(additional_fields_to_keep)

        # finally, we do the field filtering.
        self.chain(ChangeFieldsMapper(keep_fields=keep_fields))
