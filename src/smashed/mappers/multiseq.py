import itertools
import random
from typing import (
    Any,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..base import BatchedBaseMapper, SingleBaseMapper, TransformElementType


class TokensSequencesPaddingMapper(SingleBaseMapper):
    bos: List[int]
    sep: List[int]
    eos: List[int]

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        input_field: str = "input_ids",
    ) -> None:
        """Mapper that add BOS/SEP/EOS sequences of tokens.

        Args:
            tokenizer (PreTrainedTokenizerBase): Tokenizer to use for
                looking up special BOS/SEP/EOS tokens.
            input_field (str, optional): The field to add special tokens to.
                Defaults to 'input_ids'.
        """
        super().__init__(
            input_fields=[input_field], output_fields=[input_field]
        )
        self.bos, self.sep, self.eos = self._find_special_token_ids(tokenizer)

    @staticmethod
    def _find_special_token_ids(
        tokenizer: PreTrainedTokenizerBase,
    ) -> Tuple[List[int], List[int], List[int]]:
        """By default, tokenizers only know how to concatenate 2 fields
        as input; However, for our purposes, we might care about more than
        just 2. This function tries to figure out the best strategy by
        tokenizing two fake sequences and selecting beginning, mid, and
        end sequence(s) tokens."""

        bos: List[int] = []
        sep: List[int] = []
        eos: List[int] = []

        class FirstFakeSequenceSymbol(int):
            ...

        class SecondFakeSequenceSymbol(int):
            ...

        input_ids = tokenizer.build_inputs_with_special_tokens(
            [FirstFakeSequenceSymbol()], [SecondFakeSequenceSymbol()]
        )

        # anything before the first symbol goes into BOS
        seq_to_append_to = bos
        for token in input_ids:
            if isinstance(token, FirstFakeSequenceSymbol):
                # we found the first fake symbol! switch to
                # mid representation
                seq_to_append_to = sep
            elif isinstance(token, SecondFakeSequenceSymbol):
                # we found the second fake symbol! now we are
                # dealing with EOS
                seq_to_append_to = eos
            else:
                # this is a special token symbol
                seq_to_append_to.append(token)

        return bos, sep, eos

    def transform(self, data: TransformElementType) -> TransformElementType:
        sequences = data[self.input_fields[0]]
        seqs_count = len(sequences)

        padded_sequences = [
            (self.bos if i == 0 else [])
            + seq
            + (self.eos if (i + 1) == seqs_count else self.sep)
            for i, seq in enumerate(sequences)
        ]
        data[self.input_fields[0]] = padded_sequences

        return data


class AttentionMaskSequencePaddingMapper(TokensSequencesPaddingMapper):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        input_field: str = "attention_mask",
    ) -> None:
        """Mapper to add BOS/SEP/EOS tokens to an attention mask sequence.

        Args:
            tokenizer (PreTrainedTokenizerBase): Tokenizer to use for
                looking up size of special BOS/SEP/EOS tokens.
            input_field (str, optional): The field to add special tokens to.
                Defaults to 'attention_mask'.
        """
        super().__init__(tokenizer=tokenizer, input_field=input_field)

        # attention masks are always masked with ones
        self.bos = [1 for _ in self.bos]
        self.sep = [1 for _ in self.sep]
        self.eos = [1 for _ in self.eos]


class TokenTypeIdsSequencePaddingMapper(TokensSequencesPaddingMapper):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        input_field: str = "token_type_ids",
    ) -> None:
        """Mapper to add BOS/SEP/EOS tokens to a token type ids sequence.

        Args:
            tokenizer (PreTrainedTokenizerBase): Tokenizer to use for
                looking up size of special BOS/SEP/EOS tokens.
            input_field (str, optional): The field to add special tokens to.
                Defaults to 'token_type_ids'.
        """
        super().__init__(tokenizer=tokenizer, input_field=input_field)

    def transform(self, data: TransformElementType) -> TransformElementType:
        sequences = data[self.input_fields[0]]
        seqs_count = len(sequences)
        padded_sequences = [
            (
                # a sequence start with BOS tags or SEP tags
                [i for _ in self.bos]
                if i == 0
                else [i for _ in self.sep]
            )
            + seq
            + (
                # a sequence ends with EOS tags or nothing if it is not
                # the last sequence
                [i for _ in self.eos]
                if (i + 1) == seqs_count
                else []
            )
            for i, seq in enumerate(sequences)
        ]
        data[self.input_fields[0]] = padded_sequences

        return data


class MakeAttentionMaskMapper(SingleBaseMapper):
    def __init__(
        self,
        input_field: str = "input_ids",
        output_field: str = "attention_mask",
    ) -> None:
        """Mapper to create attention masks from input ids.

        Args:
            input_field (str, optional): The field to determine the
                shape of the attention mask. Defaults to 'input_ids'.
            output_field (str, optional): The name of the field containing
                the attention mask. Defaults to 'attention_mask'.
        """
        super().__init__(
            input_fields=[input_field], output_fields=[output_field]
        )

    def transform(self, data: TransformElementType) -> TransformElementType:
        sequences = data[self.input_fields[0]]
        attention_masks = [[1 for _ in range(len(seq))] for seq in sequences]
        data[self.output_fields[0]] = attention_masks
        return data


class LabelsMaskerMapper(BatchedBaseMapper):
    def __init__(
        self,
        labels_field: str = "labels",
        strategy: Literal["all", "one", "sample"] = "all",
        sample_prob: Optional[float] = None,
        label_mask_id: Union[int, float] = -100,
    ) -> None:
        """Given a sequence of labels, this mapper will mask some of them.
        Useful when wanting to create more samples by masking a subset of
        labels, or when running evaluations and want to predict one label at
        the time.

        Args:
            labels_field (str, optional): The field containing the labels.
                Defaults to 'labels'.
            strategy (Literal['all', 'one', 'sample'], optional): The strategy
                to use for masking. If 'all', no values are masked. If 'one',
                we generate m separate samples for each label, where for each
                sample, we mask one label. If 'sample', we generate sample
                a subset of labels with a probability of `sample_prob`.
                Defaults to 'all'.
            sample_prob (float, optional): The probability of sampling when
                `strategy` is 'sample'. Defaults to None. If strategy is
                'sample' and sample_prob is None, an error is raised.
            label_mask_id (LabelType, optional): The value to use for a masked
                label. Defaults to -100.

        """
        super().__init__(
            input_fields=[labels_field], output_fields=[labels_field]
        )

        if strategy not in ["all", "one", "sample"]:
            raise ValueError(f"Unknown strategy {strategy}")
        elif strategy == "random" and sample_prob is None:
            raise ValueError("no `sample_prob` provided for `random` strategy")
        elif strategy == "one" and sample_prob is not None:
            raise ValueError("Do not provide `sample_prob` for `one` strategy")

        self.strategy: str = strategy
        self.sample_prob: float = sample_prob or 0.0
        self.label_mask_id: Union[int, float] = label_mask_id

    def transform(
        self, data: Iterable[TransformElementType]
    ) -> Iterable[TransformElementType]:

        if self.strategy == "all":
            # there's no masking to do if the strategy is all!
            # `all` is provided for convenience, but it's not really a
            # transformation, just an identity function
            yield from data

        labels_field_name, *_ = self.input_fields

        for sample in data:
            labels = sample[labels_field_name]

            if self.strategy == "one":
                # make sequences of labels where only one label is
                # not masked for each sequence; the goal is to transform
                # n sequences with m active labels into n * m sequences
                # with only one active label.
                for i, _ in enumerate(labels):

                    # new labels sequence here
                    new_labels = [
                        l if i == j else self.label_mask_id
                        for j, l in enumerate(labels)
                    ]

                    new_sample = {**sample, **{labels_field_name: new_labels}}
                    yield new_sample

                    # # add the new sample you just made to the batch
                    # new_data[labels_field_name].append(new_labels)
                    # for f in other_fields_name:
                    #     new_data[f].append(data[f][sample_pos])

            if self.strategy == "sample":
                expected_slice_size = max(
                    int(len(labels) * self.sample_prob), 1
                )
                labels_pos = list(range(len(labels)))
                random.shuffle(labels_pos)

                for active_pos in (
                    labels_pos[i : i + expected_slice_size]
                    for i in range(0, len(labels_pos), expected_slice_size)
                ):
                    # new labels sequence here
                    new_labels = [
                        l if i in active_pos else self.label_mask_id
                        for i, l in enumerate(labels)
                    ]

                    # add the new sample you just made to the batch,
                    # plus all extra fields
                    new_sample = {**sample, **{labels_field_name: new_labels}}
                    yield new_sample


class MultiSequenceStriderMapper(BatchedBaseMapper):
    def __init__(
        self,
        max_stride_count: int,
        length_reference_field: str,
        max_length: Optional[int] = None,
        extra_length_per_seq: Optional[int] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_step: Optional[int] = None,
    ) -> None:
        """Mapper to create multiple subset sequences from a single sequence
        of sequences.

        The multiple sequences are created by sliding a window of size
        `max_stride_count` over the original sequence or `max_length` if
        the total number of accumulated tokens exceeds it (whichever comes
        first).

        Args:
            max_stride_count (int): The maximum number of sequences to include
                in each subset sequence of sequences.
            length_reference_field (str): The field to use to determine the
                rolling length of a subset sequence.
            max_length (int, optional): The maximum length of units in
                the subset sequence. Defaults to None (i.e., not used).
            extra_length_per_seq (int, optional): Optional field in case
                you we expect each sequence to be extended by another mapper
                after MultiSequenceStriderMapper. For example, if you are
                expecting to add special tokens for BOS/EOS/SEP,
                extra_length_per_seq could be 2 if BOS, EOS, and SEP are
                length 1.
            tokenizer (PreTrainedTokenizerBase, optional): A HuggingFace
                tokenizer to use to determine the length of BOS/EOS/SEP
                tokens. If not provided, extra_length_per_seq is used.
            max_step (int, optional): Not used at the moment.

        """
        super().__init__(
            input_fields=[length_reference_field],
            output_fields=[length_reference_field],
        )

        self.max_stride_count = max_stride_count
        self.max_length = max_length or float("inf")

        if extra_length_per_seq is None:
            if tokenizer is not None:
                # if a tokenizer is provided, we use the extra
                # length we need for each sequence is to account for
                # padding of sequences. Because we don't know if a sequence
                # will be at the beginning, middle, or end of the batch,
                # we take the max length between the various separators
                (
                    bos,
                    sep,
                    eos,
                ) = TokensSequencesPaddingMapper._find_special_token_ids(
                    tokenizer
                )
                extra_length_per_seq = 2 * max(len(bos), len(sep), len(eos))
            else:
                # if extra length is not provided, we simply assume that
                # there is not need to account for extra length that is
                # needed for special token separation (e.g. BOS/EOS symbols)
                extra_length_per_seq = 0
        self.extra_length_per_seq = extra_length_per_seq

        if max_step is not None:
            # TODO: implement max_step to support overlapping strides
            raise NotImplementedError("max_step is not supported yet")

    def transform(
        self, data: Iterable[TransformElementType]
    ) -> Iterable[TransformElementType]:

        ref_field_name, *_ = self.input_fields

        for sample in data:
            seq_pos_start = 0
            cumulative_stride_length = 0

            for seq_pos_end in range(len(sample[ref_field_name])):
                current_seq_length = (
                    len(sample[ref_field_name][seq_pos_end])
                    + self.extra_length_per_seq
                )

                if current_seq_length > self.max_length:
                    raise ValueError(
                        "Current sequence is longer than the maximum stride"
                        f"length ({current_seq_length} > {self.max_length})"
                    )

                stride_too_long = (
                    cumulative_stride_length + current_seq_length
                ) > self.max_length
                stride_has_too_many_seqs = (
                    seq_pos_end - seq_pos_start
                ) >= self.max_stride_count

                if stride_too_long or stride_has_too_many_seqs:
                    yield {
                        k: v[seq_pos_start:seq_pos_end]
                        for k, v in sample.items()
                    }

                    cumulative_stride_length = 0
                    seq_pos_start = seq_pos_end

                # now that the current sequence is included in the next
                # stride, we add its length to the cumulative stride length
                cumulative_stride_length += current_seq_length

            # yield the last sequence
            out = {k: v[seq_pos_start:] for k, v in sample.items()}

            if len(out[ref_field_name]) < 1:
                import ipdb

                ipdb.set_trace()
            yield out


class SingleValueToSequenceMapper(SingleBaseMapper):
    def __init__(
        self,
        single_value_field: str,
        like_field: str = "input_ids",
        strategy: Literal["first", "last", "all"] = "first",
        padding_id: Union[int, float] = -100,
    ) -> None:
        """Mapper to create a sequence of values from single value.
        Useful when casting a sequence classification task to a sequence
        tagging task, e.g. making a prediction for a sequence of sentences
        by concatenating the sentences, and then predicting on the BOS/SEP
        tokens.

        Args:
            single_value_field: name of the field containing the single
                label value.
            like_field: name of the field whose shape will be used to repeat
                the single value to create the sequence. Default is
                'input_ids'.
            strategy: strategy to use to create the sequence.
                - If 'first', the single value will be the first element of the
                    new sequence, and all other positions will be filled with
                    the padding_id.
                - If 'last', the single value will be the last element of the
                    new sequence, and all other positions will be filled with
                    the padding_id.
                - If 'all', the single value will be repeated for each position
                    of the new sequence; the padding_id will be ignored.
            padding_id: id to use for the padding token. Default is -100.
        """
        super().__init__(
            input_fields=[single_value_field, like_field],
            output_fields=[single_value_field],
        )
        self.strategy = strategy
        self.padding_id = padding_id

    def _make_sequence_from_value(
        self, value: Union[int, float], like_seq: Sequence[Any]
    ) -> Sequence[Union[int, float]]:

        if self.strategy == "first":
            return [value] + [
                self.padding_id for _ in range(len(like_seq) - 1)
            ]
        elif self.strategy == "last":
            return [self.padding_id for _ in range(len(like_seq) - 1)] + [
                value
            ]
        elif self.strategy == "all":
            return [value for _ in like_seq]
        else:
            raise ValueError(f"Strategy {self.strategy} is not supported")

    def transform(self, data: TransformElementType) -> TransformElementType:
        labels_field_name, like_field_name, *_ = self.input_fields

        data[labels_field_name] = [
            self._make_sequence_from_value(
                value=label, like_seq=data[like_field_name][i]
            )
            for i, label in enumerate(data[labels_field_name])
        ]
        return data


class SequencesConcatenateMapper(SingleBaseMapper):
    def __init__(self, concat_fields: Optional[List[str]] = None):
        super().__init__(
            input_fields=concat_fields, output_fields=concat_fields
        )
        self.concat_fields = (
            set(concat_fields) if concat_fields is not None else None
        )

    def _to_concat(self, field_name: str) -> bool:
        return self.concat_fields is None or field_name in self.concat_fields

    def transform(self, data: TransformElementType) -> TransformElementType:
        return {
            key: (
                list(itertools.chain.from_iterable(value))
                if self._to_concat(key)
                else value
            )
            for key, value in data.items()
        }
