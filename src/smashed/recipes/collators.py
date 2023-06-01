from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

from necessary import necessary

from ..base import BaseRecipe, SingleBaseMapper
from ..mappers import (
    FixedBatchSizeMapper,
    FromTokenizerListCollatorMapper,
    FromTokenizerTensorCollatorMapper,
    ListCollatorMapper,
    Python2TorchMapper,
    TensorCollatorMapper,
)

with necessary("transformers", soft=True) as TRANSFORMERS_AVAILABLE:
    if TRANSFORMERS_AVAILABLE or TYPE_CHECKING:
        from transformers.tokenization_utils_base import (
            PreTrainedTokenizerBase,
        )

with necessary("torch", soft=True) as PYTORCH_AVAILABLE:
    if PYTORCH_AVAILABLE or TYPE_CHECKING:
        import torch


class CollateFnMixIn(SingleBaseMapper):
    def __init__(self, do_not_collate: Optional[Sequence[str]] = None) -> None:
        self.do_not_collate = dict.fromkeys(do_not_collate or [])
        super().__init__()

    def collate(self, batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        # we extract the fields that are not to be collated; we will
        # reinsert them later

        # skip fields that do not support collation as tensors; we will
        # reinsert them later as lists in each batch
        skipped: Dict[str, List[Any]] = {
            field: [sample.pop(field) for sample in batch if field in sample]
            for field in self.do_not_collate
        }

        # collator will return a list with a single element with all the
        # data from the batch. We need to unpack it and return the value.
        out = self.map(batch)
        if len(out) != 1:
            raise ValueError(
                f"Collator returned {len(out)} elements, but expected 1; "
                "please report this as a bug."
            )
        collated_batch: Dict[str, List[Any]] = out[0]

        # here we reattach the answers to the batch
        # "if v" prevents us from adding empty lists,
        # which correspond to fields that were not present in the batch
        collated_batch.update({k: v for k, v in skipped.items() if v})

        return collated_batch

    def get_tensorizer(
        self,
        field_cast_map: Optional[
            Mapping[str, Union[str, "torch.dtype"]]
        ] = None,
        device: Optional[Union["torch.device", str]] = None,
    ) -> Python2TorchMapper:
        # this turns lists of ints/floats into tensors
        return Python2TorchMapper(field_cast_map=field_cast_map, device=device)

    def get_batcher(self, keep_last: bool) -> FixedBatchSizeMapper:
        # the collator already receives the "right" number of samples
        # in a list (that is, the batch size), so we do not need to
        # split it further; rather, the fixed size batcher will
        return FixedBatchSizeMapper(batch_size="max", keep_last=keep_last)


class CollatorRecipe(CollateFnMixIn, BaseRecipe):
    def __init__(
        self,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        do_not_collate: Optional[Sequence[str]] = None,
        keep_last: bool = True,
        pad_to_length: Optional[Union[int, Sequence[int]]] = None,
        pad_to_multiple_of: Optional[int] = None,
        fields_pad_ids: Optional[Mapping[str, int]] = None,
        unk_fields_pad_id: Optional[int] = None,
        field_cast_map: Optional[
            Mapping[str, Union[str, "torch.dtype"]]
        ] = None,
        device: Optional[Union["torch.device", str]] = None,
    ) -> None:
        """A recipe that creates a chain of mappers that can collate a sequence
        of tensors into a batch of tensors.

        Args:
            tokenizer (PretrainedTokenizerBase, optional): the tokenizer to use
                when collating. If None, the collator will assume that the
                symbols to pad with are provided in the fields_pad_ids
                argument. Defaults to None.
            do_not_collate (Sequence[str], optional): a sequence of fields
                that should not be collated but kept as lists of values.
                Defaults to None.
            keep_last (bool, optional): whether to keep the last batch if it
                is smaller than the batch size. Defaults to True.
            pad_to_length (Union[int, Sequence[int]], optional): the length
                to pad the sequences to. If an int, all sequences will be
                padded to the same length. If a sequence, the length of the
                sequence must match the number of fields in the batch. If
                None, the sequences will not be padded. Defaults to None.
            pad_to_multiple_of (int, optional): the length to pad the
                sequences to. If None, the sequences will not be padded or
                padded according to the pad_to_length argument. Defaults to
                None.
            fields_pad_ids (Mapping[str, int], optional): a mapping from
                field names to the ids to use for padding. If None, the
                collator will assume that the symbols to pad with are
                provided in the fields_pad_ids argument. Defaults to None.
            unk_fields_pad_id (int, optional): the id to use for padding
                unknown tokens. If None, the collator will assume that the
                symbols to pad with are provided in the fields_pad_ids
                argument. Defaults to None.
            field_cast_map (Mapping[str, Union[str, torch.dtype]], optional):
                a mapping from field names to the type to cast the field to.
                If None, the collator will use the default type inferred by
                torch.tensor. Defaults to None.
            device (Union[torch.device, str], optional): the device to
                create the tensors on. If None, the collator will use the
                default device. Defaults to None.
        """

        super().__init__(do_not_collate=do_not_collate)

        self.chain(
            self.get_tensorizer(field_cast_map=field_cast_map, device=device)
        )
        self.chain(self.get_batcher(keep_last=keep_last))

        if tokenizer:
            self.chain(
                FromTokenizerTensorCollatorMapper(
                    tokenizer=tokenizer,
                    fields_pad_ids=(fields_pad_ids or {}),
                    unk_fields_pad_id=unk_fields_pad_id,
                    pad_to_length=pad_to_length,
                    pad_to_multiple_of=pad_to_multiple_of,
                )
            )
        else:
            if fields_pad_ids is None:
                raise ValueError(
                    "fields_pad_ids must be provided when no tokenizer!"
                )
            self.chain(
                TensorCollatorMapper(
                    fields_pad_ids=fields_pad_ids,
                    unk_fields_pad_id=unk_fields_pad_id,
                    pad_to_length=pad_to_length,
                )
            )


class SlowCollatorRecipe(CollateFnMixIn, BaseRecipe):
    def __init__(
        self,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        do_not_collate: Optional[Sequence[str]] = None,
        keep_last: bool = True,
        pad_to_length: Optional[Union[int, Sequence[int]]] = None,
        pad_to_multiple_of: Optional[int] = None,
        fields_pad_ids: Optional[Mapping[str, int]] = None,
        unk_fields_pad_id: Optional[int] = None,
    ) -> None:
        """A recipe that creates a chain of mappers that can collate a sequence
        of lists into a batch of tensors. It is slower than the CollatorRecipe,
        as it pads Python lists before converting them to tensors.

                Args:
            tokenizer (PretrainedTokenizerBase, optional): the tokenizer to use
                when collating. If None, the collator will assume that the
                symbols to pad with are provided in the fields_pad_ids
                argument. Defaults to None.
            do_not_collate (Sequence[str], optional): a sequence of fields
                that should not be collated but kept as lists of values.
                Defaults to None.
            keep_last (bool, optional): whether to keep the last batch if it
                is smaller than the batch size. Defaults to True.
            pad_to_length (Union[int, Sequence[int]], optional): the length
                to pad the sequences to. If an int, all sequences will be
                padded to the same length. If a sequence, the length of the
                sequence must match the number of fields in the batch. If
                None, the sequences will not be padded. Defaults to None.
            pad_to_multiple_of (int, optional): the length to pad the
                sequences to. If None, the sequences will not be padded or
                padded according to the pad_to_length argument. Defaults to
                None.
            fields_pad_ids (Mapping[str, int], optional): a mapping from
                field names to the ids to use for padding. If None, the
                collator will assume that the symbols to pad with are
                provided in the fields_pad_ids argument. Defaults to None.
            unk_fields_pad_id (int, optional): the id to use for padding
                unknown tokens. If None, the collator will assume that the
                symbols to pad with are provided in the fields_pad_ids
                argument. Defaults to None.
            field_cast_map (Mapping[str, Union[str, torch.dtype]], optional):
                a mapping from field names to the type to cast the field to.
                If None, the collator will use the default type inferred by
                torch.tensor. Defaults to None.
            device (Union[torch.device, str], optional): the device to
                create the tensors on. If None, the collator will use the
                default device. Defaults to None.
        """
        super().__init__(do_not_collate=do_not_collate)

        self.chain(self.get_batcher(keep_last=keep_last))

        if tokenizer:
            self.chain(
                FromTokenizerListCollatorMapper(
                    tokenizer=tokenizer,
                    fields_pad_ids=(fields_pad_ids or {}),
                    unk_fields_pad_id=unk_fields_pad_id,
                    pad_to_length=pad_to_length,
                    pad_to_multiple_of=pad_to_multiple_of,
                )
            )
        else:
            if fields_pad_ids is None:
                raise ValueError(
                    "fields_pad_ids must be provided when no tokenizer!"
                )
            self.chain(
                ListCollatorMapper(
                    fields_pad_ids=fields_pad_ids,
                    unk_fields_pad_id=unk_fields_pad_id,
                    pad_to_length=pad_to_length,
                )
            )

        self.chain(self.get_tensorizer())
