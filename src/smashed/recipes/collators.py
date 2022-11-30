from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..base import BaseRecipe, SingleBaseMapper
from ..mappers import (
    FixedBatchSizeMapper,
    FromTokenizerListCollatorMapper,
    FromTokenizerTensorCollatorMapper,
    ListCollatorMapper,
    Python2TorchMapper,
    TensorCollatorMapper,
)


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

    def get_tensorizer(self) -> Python2TorchMapper:
        # this turns lists of ints/floats into tensors
        return Python2TorchMapper()

    def get_batcher(self, keep_last: bool) -> FixedBatchSizeMapper:
        # the collator already receives the "right" number of samples
        # in a list (that is, the batch size), so we do not need to
        # split it further; rather, the fixed size batcher will
        return FixedBatchSizeMapper(batch_size="max", keep_last=keep_last)


class CollatorRecipe(CollateFnMixIn, BaseRecipe):
    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        do_not_collate: Optional[Sequence[str]] = None,
        keep_last: bool = True,
        pad_to_length: Optional[Union[int, Sequence[int]]] = None,
        fields_pad_ids: Optional[Mapping[str, int]] = None,
        unk_fields_pad_id: Optional[int] = None,
    ) -> None:
        super().__init__(do_not_collate=do_not_collate)

        self.chain(self.get_tensorizer())
        self.chain(self.get_batcher(keep_last=keep_last))

        if tokenizer:
            self.chain(
                FromTokenizerTensorCollatorMapper(
                    tokenizer=tokenizer,
                    fields_pad_ids=(fields_pad_ids or {}),
                    unk_fields_pad_id=unk_fields_pad_id,
                    pad_to_length=pad_to_length,
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
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        do_not_collate: Optional[Sequence[str]] = None,
        keep_last: bool = True,
        pad_to_length: Optional[Union[int, Sequence[int]]] = None,
        fields_pad_ids: Optional[Mapping[str, int]] = None,
        unk_fields_pad_id: Optional[int] = None,
    ) -> None:
        super().__init__(do_not_collate=do_not_collate)

        self.chain(self.get_batcher(keep_last=keep_last))

        if tokenizer:
            self.chain(
                FromTokenizerListCollatorMapper(
                    tokenizer=tokenizer,
                    fields_pad_ids=(fields_pad_ids or {}),
                    unk_fields_pad_id=unk_fields_pad_id,
                    pad_to_length=pad_to_length,
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
