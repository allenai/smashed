import functools
from collections import abc
from itertools import chain
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..base import SingleBaseMapper, TransformElementType
from ..base.abstract import AbstractBaseMapper

__all__ = [
    "ListCollatorMapper",
    "TensorCollatorMapper",
    "FromTokenizerListCollatorMapper",
    "FromTokenizerTensorCollatorMapper",
]


class BaseCollator(AbstractBaseMapper):
    def __init__(
        self,
        pad_to_length: Optional[Union[int, Sequence[int]]] = None,
        fields_pad_ids: Optional[Mapping[str, int]] = None,
        unk_fields_pad_id: Optional[int] = None,
    ):
        """Create a collator.

        Args:
            pad_to_length (Union[int, Sequence[int]], optional): If provided
                and is single value, pad all sequences to this length. If
                provided  and is sequence, we assume we should pad each
                dimension to the  corresponding length. In both cases, we will
                raise an error if any sequence is longer than the requested
                length. When not provided or None, sequences will be padded to
                the length of the longest sequence. Defaults to None.
            fields_pad_ids (Mapping[str, int], optional): A mapping from field
                names to the padding value to use for that field. If not
                provided, the mapper will fail unless the unk_fields_pad_id
                attribute is set.
            unk_fields_pad_id (int, optional): The padding value to use for
                any field that is not in fields_pad_ids. If not provided, an
                error will be raised if a field is not in fields_pad_ids.
        """
        self.fields_pad_ids = fields_pad_ids or {}
        self.pad_to_length = pad_to_length
        self.unk_fields_pad_id = unk_fields_pad_id

        if self.unk_fields_pad_id is None and self.fields_pad_ids is None:
            raise ValueError(
                "Either `unk_fields_pad_id` or `fields_to_pad` "
                "must be provided"
            )

        super().__init__()

    def _get_padding_value(self, field_name: str) -> int:
        if field_name in self.fields_pad_ids:
            return self.fields_pad_ids[field_name]
        elif self.unk_fields_pad_id is not None:
            return self.unk_fields_pad_id
        else:
            raise ValueError(
                f"Must specify a padding value for field {field_name}"
                f"or provide a extra_fields_padding_id attribute to "
                "the mapper to handle unrecognized fields"
            )

    def collate(
        self, batch: Sequence[TransformElementType]
    ) -> TransformElementType:
        """This method complies with PyTorch's DataLoader interface."""
        return self.transform(
            {k: [d[k] for d in batch] for k in batch[0].keys()}
        )


# Alias for backwards compatibility
CollatorMixIn = BaseCollator


class FromTokenizerMixIn(BaseCollator):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pad_to_length: Optional[Union[int, Sequence[int]]] = None,
        fields_pad_ids: Optional[Mapping[str, int]] = None,
        unk_fields_pad_id: Optional[int] = None,
    ):
        """Create a collator for tensors.

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to use to pad
                common fields.
            pad_to_length (Union[int, Sequence[int]], optional): If provided
                and is single value, pad all sequences to this length. If
                provided  and is sequence, we assume we should pad each
                dimension to the  corresponding length. In both cases, we will
                raise an error if any sequence is longer than the requested
                length. When not provided or None, sequences will be padded to
                the length of the longest sequence. Defaults to None.
            fields_pad_ids (Mapping[str, int], optional): A mapping from field
                names to the padding value to use for that field. If not
                provided, the mapper will fail unless the unk_fields_pad_id
                attribute is set.
            unk_fields_pad_id (int, optional): The padding value to use for
                any field that is not in fields_pad_ids. If not provided, an
                error will be raised if a field is not in fields_pad_ids.
        """

        fields_pad_ids = {
            "input_ids": tokenizer.pad_token_id or 0,
            "attention_mask": 0,
            "token_type_ids": tokenizer.pad_token_type_id or 0,
            "overflow_to_sample_mapping": 0,
            "special_tokens_mask": 0,
            "offset_mapping": 0,
            "length": 0,
            **(fields_pad_ids or {}),
        }
        super().__init__(
            pad_to_length=pad_to_length,
            unk_fields_pad_id=unk_fields_pad_id,
            fields_pad_ids=fields_pad_ids,
        )


class TensorCollatorMapper(BaseCollator, SingleBaseMapper):
    """
    A collator mapper that collates sequences of n tensors into a single
    tensor of shape (n, ...) where ... is the maximum size of the
    sequences. Uses the values passed to fields_pad_ids to determine
    how to pad each field.

    If used with a Pytorch DataLoader, the collator mapper must be
    called as follows:

    >>> collator = TensorsCollatorMapper(...)
    >>> data_loader = DataLoader(..., collate_fn=collator.transform)
    """

    @staticmethod
    def _pad(
        sequence: Sequence[torch.Tensor],
        pad_value: int,
        dim: int = 0,
        pad_to_length: Optional[Union[int, Sequence[int]]] = None,
    ) -> torch.Tensor:

        # make sure type of input is right
        if not (
            isinstance(sequence, abc.Sequence)
            and all(isinstance(elem, torch.Tensor) for elem in sequence)
        ):
            raise ValueError(
                "Each element to collate must be a sequence of torch.Tensor, "
                f"not {type(sequence)}"
            )

        # the `view` is because we need to add a new dimension as the
        # dimension alongside which we batch
        sequence = [torch.unsqueeze(tensor, dim=dim) for tensor in sequence]

        # this contains maximum length of all the sequences
        max_lengths = [max(t) for t in zip(*(t.size() for t in sequence))]

        if isinstance(pad_to_length, int):
            # if pad_to_length is a single integer, we pad all sequences to
            # the same length
            pad_to_length = [pad_to_length] * len(sequence)

        # if pad_to_length is provided, we need to make sure that
        # each dimension does not exceed the pad_to_length[i].
        if pad_to_length is not None:
            for i in range(len(max_lengths)):
                if i == dim:
                    # do not touch the dimension we are batching along
                    continue
                elif max_lengths[i] > pad_to_length[i]:
                    # raise an error if the pad_to_length requested
                    # for this dimension is smaller than the maximum
                    # length of the current sequence of this dimension
                    raise ValueError(
                        f"Tried to pad sequence to length {pad_to_length} "
                        f"but sequence had shape {max_lengths}"
                    )
                else:
                    max_lengths[i] = pad_to_length[i]

        # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad
        # according to that page, we need to create pad shapes is reverse.
        pad_shapes = tuple(
            tuple(
                chain.from_iterable(
                    (0, m - s)
                    for s, m in zip(t.size()[::-1], max_lengths[::-1])
                )
            )
            # we do padding shapes for each tensor
            for t in sequence
        )
        # call each pad on each of the tensors with the appropriate padding
        to_stack = tuple(
            torch.nn.functional.pad(
                tensor, pad, mode="constant", value=pad_value
            )
            for tensor, pad in zip(sequence, pad_shapes)
        )

        return torch.cat(to_stack, dim=dim)

    def transform(
        self: "TensorCollatorMapper", data: Dict[str, Sequence[torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:

        collated_data = {
            field_name: self._pad(
                sequence=list_of_tensors,
                pad_value=self._get_padding_value(field_name=field_name),
                pad_to_length=self.pad_to_length,
            )
            for field_name, list_of_tensors in data.items()
        }
        return collated_data


class FromTokenizerTensorCollatorMapper(
    FromTokenizerMixIn, TensorCollatorMapper
):
    """
    A collator mapper that collates sequences of n tensors into a
    single tensor of shape (n, ...) where ... is the maximum size of the
    sequences. Uses the provided tokenizer to determine how to pad common
    fields for NLP tasks, such as `input_ids`, `attention_mask`,
    `token_type_ids`, etc. Padding values for further fields can be
    provided in the `fields_pad_ids` argument.

    If used with a Pytorch DataLoader, the collator mapper must be
    called as follows:
    >>> tokenizer = AutoTokenizer.from_pretrained(...)
    >>> collator = FromTokenizerTensorsCollatorMapper(tokenizer, ...)
    >>> data_loader = DataLoader(..., collate_fn=collator.transform)
    """


class ListCollatorMapper(BaseCollator, SingleBaseMapper):
    """Given a `sequence_to_pad` dict of comprised of <field names, value to
    pad field with>, figures out how much to pad them such that all examples in
    dataset in those fields have same length. This mapper can be useful for
    cases where HuggingFace tokenizer padding behaves weirdly, which is the
    case when it is customized to produce extra keys. For example,

    tokenizer.pad([{'input_ids': [0, 1, 2], 'aaa': [3, 3, 3]},
                   {'input_ids': [3, 4], 'aaa': [4, 4]}])

    will correctly pad `input_ids`, but not `aaa`.  This can break collation
    because it requires calls to `tokenizer.pad`.
    """

    pad_to_length: int

    def _get_list_shape_recursive(
        self, sequence: Sequence[Any]
    ) -> Tuple[int, ...]:
        if not isinstance(sequence, abc.Sequence):
            # we have unpacked as far as we can go; this is just a
            # single element.
            return tuple()

        # this iterator will yield the shape of each element in the sequence
        inner_dims = (self._get_list_shape_recursive(s) for s in sequence)

        # the acutal shape is the maximum of the inner dims
        inner_shape = tuple(max(dims) for dims in zip(*inner_dims))

        return (len(sequence), *inner_shape)

    def _pad_recursive(
        self, sequence: List[Any], shape: Sequence[int], padding_symbol: Any
    ) -> List[Any]:
        """Recursively pads a list of [lists, ...].

        Args:
            sequence (List[Any]): The list to pad.
            shape (Sequence[int]): The shape to pad to.
            padding_symbol (Any): The symbol to pad with.

        Returns:
            List[Any]: _description_
        """

        if len(shape) < 2:
            # we have reached the end of the shape; this is a single element
            return sequence

        _, dim_to_pad_shape, *rest_shape = shape

        nested_pad_symbol = functools.reduce(
            lambda x, _: [x], range(len(rest_shape)), padding_symbol
        )

        # Let's walk through the nested padding process here.
        #
        # Our overall goal is to turn a potentially nested sequence of
        # list such as:
        #
        #       [[[1 2 3] [4 5]    ]
        #        [[6 7]   [8]   [9]]]
        #
        # Into a list of lists of the same length, such as:
        #
        #       [[[1 2 3] [4 5 0] [0 0 0]]
        #        [[6 7 0] [8 0 0] [9 0 0]]]
        #
        # The first step is to add any sequence that is completely missing
        # such as the last example of the first row above. That will get us
        # the following:
        #
        #      [[[1 2 3] [4 5] [0]]
        #       [[6 7]   [8]   [9]]]
        #
        # We do that in the following line:
        sequence_with_brand_new_padding = (
            sub_seq + [nested_pad_symbol] * (dim_to_pad_shape - len(sub_seq))
            for sub_seq in sequence
        )

        # The second step is to recursively pad the inner lists by
        # calling this function on each subsequence. We do that as follows:
        padded_sequence = [
            self._pad_recursive(
                sequence=sub_seq,
                shape=(dim_to_pad_shape, *rest_shape),
                padding_symbol=padding_symbol,
            )
            for sub_seq in sequence_with_brand_new_padding
        ]

        return padded_sequence

    def _pad(
        self: "ListCollatorMapper",
        seq_of_seq_to_pad: List[Any],
        padding_symbol: Any,
    ) -> List[Any]:

        padding_shape = self._get_list_shape_recursive(seq_of_seq_to_pad)

        if self.pad_to_length is not None:
            if not all(p <= self.pad_to_length for p in padding_shape):
                raise ValueError(
                    "PaddingMapper expects every input sequence to be less"
                    "than or equal to the `pad_to_length`. Please handle"
                    "any truncation or whatever upstream in a different"
                    " mapper, such as TokenizerMapper."
                    f"\t{padding_shape} > {self.pad_to_length}"
                    f"\t{seq_of_seq_to_pad}"
                )
            padding_shape = (self.pad_to_length,) * len(padding_shape)

        if len(padding_shape) < 2:
            # nothing to pad here; we need at minimum a list of lists
            # for padding to make any sense.
            return seq_of_seq_to_pad

        padded_sequence = self._pad_recursive(
            sequence=seq_of_seq_to_pad,
            shape=padding_shape,
            padding_symbol=padding_symbol,
        )
        return padded_sequence

    def transform(self, data: TransformElementType) -> TransformElementType:
        """Add padding to all list elements for the fields we specify."""

        return {
            field_name: self._pad(
                seq_of_seq_to_pad=field_value,
                padding_symbol=self._get_padding_value(field_name=field_name),
            )
            for field_name, field_value in data.items()
        }


class FromTokenizerListCollatorMapper(FromTokenizerMixIn, ListCollatorMapper):
    """Performs collation of a list to a given length. Uses the provided
    tokenizer to determine how to pad common fields for NLP tasks, such as
    `input_ids`, `attention_mask`, `token_type_ids`, etc. Padding values for
    further fields can be provided in the `fields_pad_ids` argument."""
