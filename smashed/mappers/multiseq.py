import itertools
import random
from typing import (Any, Iterable, List, Literal, Optional,
                    Sequence, Tuple, TypeVar)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..base import BaseMapper, TransformElementType


LabelType = TypeVar('LabelType', int, float)


class TokensSequencesPaddingMapper(BaseMapper):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        input_field: str = 'input_ids',
    ) -> None:
        super().__init__()

        self.input_fields = [input_field]
        self.output_fields = [input_field]
        self.bos, self.sep, self.eos = self._find_special_token_ids(tokenizer)
        self.batched = False

    @staticmethod
    def _find_special_token_ids(
        tokenizer: PreTrainedTokenizerBase
    ) -> Tuple[Sequence[int], Sequence[int], Sequence[int]]:
        """By default, tokenizers only know how to concatenate 2 fields
        as input; However, for our purposes, we might care about more than
        just 2. This function tries to figure out the best strategy by
        tokenizing two fake sequences and selecting beginning, mid, and
        end sequence(s) tokens."""

        bos, sep, eos = [], [], []

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
            (self.bos if i == 0 else []) +
            seq +
            (self.eos if (i + 1) == seqs_count else self.sep)
            for i, seq in enumerate(sequences)
        ]
        data[self.input_fields[0]] = padded_sequences

        return data


class AttentionMaskSequencePaddingMapper(TokensSequencesPaddingMapper):
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 input_field: str = 'attention_mask') -> None:
        super().__init__(tokenizer=tokenizer, input_field=input_field)

        # attention masks are always masked with ones
        self.bos = [1 for _ in self.bos]
        self.sep = [1 for _ in self.sep]
        self.eos = [1 for _ in self.eos]


class TokenTypeIdsSequencePaddingMapper(TokensSequencesPaddingMapper):
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 input_field: str = 'token_type_ids') -> None:
        super().__init__(tokenizer=tokenizer, input_field=input_field)

    def transform(self, data: TransformElementType) -> TransformElementType:
        sequences = data[self.input_fields[0]]
        seqs_count = len(sequences)
        padded_sequences = [
            # a sequence start with BOS tags or SEP tags
            ([i for _ in self.bos] if i == 0 else [i for _ in self.sep]) +
            seq +
            # a sequence ends with EOS tags or nothing if it is not
            # the last sequence
            ([i for _ in self.eos] if (i + 1) == seqs_count else [])
            for i, seq in enumerate(sequences)
        ]
        data[self.input_fields[0]] = padded_sequences

        return data


class MakeAttentionMaskMapper(BaseMapper):
    def __init__(self,
                 input_field: str = 'input_ids',
                 output_field: str = 'attention_mask') -> None:
        super().__init__()
        self.input_fields = [input_field]
        self.output_fields = [output_field]

    def transform(self, data: TransformElementType) -> TransformElementType:
        sequences = data[self.input_fields[0]]
        attention_masks = [[1 for _ in range(len(seq))] for seq in sequences]
        data[self.output_fields[0]] = attention_masks
        return data


class LabelsMaskerMapper(BaseMapper):
    def __init__(self,
                 labels_field: str = 'labels',
                 strategy: Literal['all', 'one', 'sample'] = 'all',
                 sample_prob: Optional[float] = None,
                 label_mask_id: LabelType = -100) -> None:
        super().__init__()
        self.input_fields = [labels_field]
        self.output_fields = [labels_field]
        self.batched = True

        if strategy not in ['all', 'one', 'sample']:
            raise ValueError(f'Unknown strategy {strategy}')
        elif strategy == 'random' and sample_prob is None:
            raise ValueError('no `sample_prob` provided for `random` strategy')
        elif strategy == 'one' and sample_prob is not None:
            raise ValueError('Do not provide `sample_prob` for `one` strategy')

        self.strategy: str = strategy
        self.sample_prob: float = sample_prob or 0.0
        self.label_mask_id: LabelType = label_mask_id

    def transform(
        self,
        data: Iterable[TransformElementType]
    ) -> Iterable[TransformElementType]:

        if self.strategy == 'all':
            # there's no masking to do if the strategy is all!
            # `all` is provided for convenience, but it's not really a
            # transformation, just an identity function
            yield from data

        labels_field_name, *_ = self.input_fields

        for sample in data:
            labels = sample[labels_field_name]

            if self.strategy == 'one':
                # make sequences of labels where only one label is
                # not masked for each sequence; the goal is to transform
                # n sequences with m active labels into n * m sequences
                # with only one active label.
                for i, _ in enumerate(labels):

                    # new labels sequence here
                    new_labels = [l if i == j else self.label_mask_id
                                  for j, l in enumerate(labels)]

                    new_sample = {**sample,
                                  **{labels_field_name: new_labels}}
                    yield new_sample

                    # # add the new sample you just made to the batch
                    # new_data[labels_field_name].append(new_labels)
                    # for f in other_fields_name:
                    #     new_data[f].append(data[f][sample_pos])

            if self.strategy == 'sample':
                expected_slice_size = max(
                    int(len(labels) * self.sample_prob), 1
                )
                labels_pos = list(range(len(labels)))
                random.shuffle(labels_pos)

                for active_pos in (
                    labels_pos[i:i + expected_slice_size]
                    for i in range(0, len(labels_pos), expected_slice_size)
                ):
                    # new labels sequence here
                    new_labels = [l if i in active_pos else self.label_mask_id
                                  for i, l in enumerate(labels)]

                    # add the new sample you just made to the batch,
                    # plus all extra fields
                    new_sample = {**sample,
                                  **{labels_field_name: new_labels}}
                    yield new_sample


class MultiSequenceStriderMapper(BaseMapper):
    def __init__(self,
                 max_stride_count: int,
                 length_reference_field: str,
                 max_length: Optional[int] = None,
                 extra_length_per_seq: Optional[int] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 max_step: Optional[int] = None) -> None:
        super().__init__()

        self.input_fields = self.output_fields = [length_reference_field]
        self.batched = True
        self.max_stride_count = max_stride_count
        self.max_length = max_length or float('inf')

        if extra_length_per_seq is None:
            if tokenizer is not None:
                # if a tokenizer is provided, we use the extra
                # length we need for each sequence is to account for
                # padding of sequences. Because we don't know if a sequence
                # will be at the beginning, middle, or end of the batch,
                # we take the max length between the various separators
                bos, sep, eos = TokensSequencesPaddingMapper.\
                    _find_special_token_ids(tokenizer)
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
        self,
        data: Iterable[TransformElementType]
    ) -> Iterable[TransformElementType]:

        ref_field_name, *_ = self.input_fields

        for sample in data:
            seq_pos_start = 0
            cumulative_stride_length = 0

            for seq_pos_end in range(len(sample[ref_field_name])):
                current_seq_length = (
                    len(sample[ref_field_name][seq_pos_end]) +
                    self.extra_length_per_seq
                )

                if current_seq_length > self.max_length:
                    raise ValueError(
                        "Current sequence is longer than the maximum stride"
                        f"length ({current_seq_length} > {self.max_length})"
                    )

                stride_too_long = (
                    (cumulative_stride_length + current_seq_length)
                    > self.max_length
                )
                stride_has_too_many_seqs = (
                    (seq_pos_end - seq_pos_start)
                    >= self.max_stride_count
                )

                if stride_too_long or stride_has_too_many_seqs:
                    yield {k: v[seq_pos_start:seq_pos_end]
                           for k, v in sample.items()}

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


class SingleValueToSequenceMapper(BaseMapper):
    def __init__(self,
                 single_value_field: str,
                 like_field: str = 'input_ids',
                 strategy: Literal['first', 'last', 'all'] = 'first',
                 padding_id: LabelType = -100) -> None:
        super().__init__()

        self.input_fields = [single_value_field, like_field]
        self.output_fields = [single_value_field]
        self.strategy = strategy
        self.padding_id = padding_id
        self.batched = False

    def _make_sequence_from_value(
        self,
        value: LabelType,
        like_seq: Sequence[Any]
    ) -> Sequence[LabelType]:

        if self.strategy == 'first':
            return ([value] +       # type: ignore
                    [self.padding_id for _ in range(len(like_seq) - 1)])
        elif self.strategy == 'last':
            return ([self.padding_id for _ in range(len(like_seq) - 1)] +
                    [value])        # type: ignore
        elif self.strategy == 'all':
            return [value for _ in like_seq]
        else:
            raise ValueError(f'Strategy {self.strategy} is not supported')

    def transform(self, data: TransformElementType) -> TransformElementType:
        labels_field_name, like_field_name, *_ = self.input_fields

        data[labels_field_name] = [
            self._make_sequence_from_value(
                value=label, like_seq=data[like_field_name][i]
            ) for i, label in enumerate(data[labels_field_name])
        ]
        return data


class SequencesConcatenateMapper(BaseMapper):
    def __init__(self, concat_fields: Optional[List[str]] = None):
        super().__init__()

        self.concat_fields = (
            set(concat_fields) if concat_fields is not None else None
        )
        self.input_fields = concat_fields or []
        self.output_fields = concat_fields or []
        self.batched = False

    def _to_concat(self, field_name: str) -> bool:
        return self.concat_fields is None or field_name in self.concat_fields

    def transform(self, data: TransformElementType) -> TransformElementType:
        return {key: (list(itertools.chain.from_iterable(value))
                      if self._to_concat(key) else value)
                for key, value in data.items()}
