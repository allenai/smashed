from collections import abc
from itertools import chain
from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..base.mapper import SingleBaseMapper


class CollatorMapper(SingleBaseMapper):
    def __init__(
        self: "CollatorMapper",
        fields_pad_ids: Optional[Mapping[str, int]] = None,
        unk_fields_pad_id: Optional[int] = None,
    ):
        """A collator mapper that collates sequences of n tensors into a single
        tensor of shape (n, ...) where ... is the maximum size of the
        sequences. Uses the values passed to fields_pad_ids to determine
        how to pad each field.

        If used with a Pytorch DataLoader, the collator mapper must be
        called as follows:

        >>> collator = CollatorMapper(...)
        >>> data_loader = DataLoader(..., collate_fn=collator.transform)

        Args:
            fields_pad_ids (Mapping[str, int], optional): A mapping from field
                names to the padding value to use for that field. If not
                provided, the mapper will fail unless the unk_fields_pad_id
                attribute is set.
            unk_fields_pad_id (int, optional): The padding value to use for
                any field that is not in fields_pad_ids. If not provided, an
                error will be raised if a field is not in fields_pad_ids.
        """
        self.fields_pad_ids = fields_pad_ids or {}
        self.unk_fields_pad_id = unk_fields_pad_id
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

    @staticmethod
    def _pad(
        sequence: Sequence[torch.Tensor], pad_value: int, dim: int = 0
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

        max_lengths = tuple(max(t) for t in zip(*(t.size() for t in sequence)))

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
            F.pad(tensor, pad, mode="constant", value=pad_value)
            for tensor, pad in zip(sequence, pad_shapes)
        )

        return torch.cat(to_stack, dim=dim)

    def transform(
        self: "CollatorMapper", data: Dict[str, Sequence[torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:

        collated_data = {
            field_name: self._pad(
                sequence=list_of_tensors,
                pad_value=self._get_padding_value(field_name),
            )
            for field_name, list_of_tensors in data.items()
        }
        return collated_data


class FromTokenizerCollatorMapper(CollatorMapper):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        fields_pad_ids: Optional[Mapping[str, int]] = None,
        unk_fields_pad_id: Optional[int] = None,
    ):
        """A collator mapper that collates sequences of n tensors into a
        single tensor of shape (n, ...) where ... is the maximum size of the
        sequences. Uses the provided tokenizer to determine how to pad common
        fields for NLP tasks, such as `input_ids`, `attention_mask`,
        `token_type_ids`, etc. Padding values for further fields can be
        provided in the `fields_pad_ids` argument.

        If used with a Pytorch DataLoader, the collator mapper must be
        called as follows:
        >>> tokenizer = AutoTokenizer.from_pretrained(...)
        >>> collator = FromTokenizerCollatorMapper(tokenizer, ...)
        >>> data_loader = DataLoader(..., collate_fn=collator.transform)

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to use to pad
                common fields.
            fields_pad_ids (Mapping[str, int], optional): A mapping from field
                names to the padding value to use for that field.
            unk_fields_pad_id (int, optional): The padding value to use for
                any field that is not recognized. If not provided, an error
                will be raised if a field is not recognized.
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
            unk_fields_pad_id=unk_fields_pad_id,
            fields_pad_ids=fields_pad_ids,
        )
