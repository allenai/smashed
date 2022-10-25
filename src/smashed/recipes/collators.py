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

    def collate(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        # we extract the fields that are not to be collated; we will
        # reinsert them later
        skipped = {
            [b.pop(field, []) for b in batch[field]]
            for field in self.do_not_collate
        }

        # collator will return a list with a single element with all the
        # data from the batch. We need to unpack it and return the value.
        collated_batch, *_ = self.map(batch)

        # here we reattach the answers to the batch
        collated_batch.update(skipped)

        return collated_batch


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

        fields_pad_ids = fields_pad_ids or {}

        tensorizer = Python2TorchMapper()
        batcher = FixedBatchSizeMapper(batch_size="max", keep_last=keep_last)

        collator: TensorCollatorMapper
        if tokenizer:
            collator = FromTokenizerTensorCollatorMapper(
                tokenizer=tokenizer,
                fields_pad_ids=fields_pad_ids,
                unk_fields_pad_id=unk_fields_pad_id,
                pad_to_length=pad_to_length,
            )
        else:
            collator = TensorCollatorMapper(
                fields_pad_ids=fields_pad_ids,
                unk_fields_pad_id=unk_fields_pad_id,
                pad_to_length=pad_to_length,
            )

        self.chain(tensorizer >> batcher >> collator)


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

        fields_pad_ids = fields_pad_ids or {}

        tensorizer = Python2TorchMapper()
        batcher = FixedBatchSizeMapper(batch_size="max", keep_last=keep_last)

        collator: ListCollatorMapper
        if tokenizer:
            collator = FromTokenizerListCollatorMapper(
                tokenizer=tokenizer,
                fields_pad_ids=fields_pad_ids,
                unk_fields_pad_id=unk_fields_pad_id,
                pad_to_length=pad_to_length,
            )
        else:
            collator = ListCollatorMapper(
                fields_pad_ids=fields_pad_ids,
                unk_fields_pad_id=unk_fields_pad_id,
                pad_to_length=pad_to_length,
            )

        self.chain(batcher >> collator >> tensorizer)
