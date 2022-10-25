from typing import Mapping, Optional, Sequence, Union

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..base.recipes import BaseRecipe
from ..mappers import (
    FixedBatchSizeMapper,
    FromTokenizerListCollatorMapper,
    FromTokenizerTensorCollatorMapper,
    ListCollatorMapper,
    Python2TorchMapper,
    TensorCollatorMapper,
)


class CollatorRecipe(BaseRecipe):
    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        keep_last: bool = True,
        pad_to_length: Optional[Union[int, Sequence[int]]] = None,
        fields_pad_ids: Optional[Mapping[str, int]] = None,
        unk_fields_pad_id: Optional[int] = None,
    ) -> None:
        super().__init__()

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


class ReverseCollatorRecipe(BaseRecipe):
    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        keep_last: bool = True,
        pad_to_length: Optional[Union[int, Sequence[int]]] = None,
        fields_pad_ids: Optional[Mapping[str, int]] = None,
        unk_fields_pad_id: Optional[int] = None,
    ) -> None:
        super().__init__()

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
