from abc import ABCMeta
from typing import Any, Dict, Iterable, List, Tuple, TypeVar

import torch
from necessary import necessary

from ..base.mapper import (
    BatchedBaseMapper,
    DatasetInterfaceMapper,
    SingleBaseMapper,
)
from ..base.types import TransformBatchType
from ..mappers import (
    batchers,
    collators,
    converters,
    fields,
    multiseq,
    shape,
    tokenize,
)
from ..mappers.contrib import sse

with necessary("datasets", soft=True):
    from datasets import features
    from datasets.arrow_dataset import Dataset
    from datasets.iterable_dataset import IterableDataset

HfDatasetType = TypeVar("HfDatasetType", Dataset, IterableDataset)


__all__ = [
    "TokensSequencesPaddingMapper",
    "AttentionMaskSequencePaddingMapper",
    "TokenTypeIdsSequencePaddingMapper",
    "MakeAttentionMaskMapper",
    "LabelsMaskerMapper",
    "MultiSequenceStriderMapper",
    "SingleValueToSequenceMapper",
    "SequencesConcatenateMapper",
    "FlattenMapper",
    "BinarizerMapper",
    "TokenizerMapper",
    "OneVsOtherAnnotatorMapper",
    "ChangeFieldsMapper",
    "ValidUnicodeMapper",
    "FixedBatchSizeMapper",
    "ListCollatorMapper",
    "TensorCollatorMapper",
    "FromTokenizerListCollatorMapper",
    "FromTokenizerTensorCollatorMapper",
    "Python2TorchMapper",
    "Torch2PythonMapper",
]


class HuggingFaceDatasetsInterfaceMapper(
    DatasetInterfaceMapper, metaclass=ABCMeta
):
    def _batch_transform(
        self: "HuggingFaceDatasetsInterfaceMapper", data: TransformBatchType
    ) -> TransformBatchType:
        """Unrolls a datasets.Dataset batch, which is a dictionary of
        <features, list of feature values for each sample> into a iterable of
        dictionaries that can be passed to the transform function."""

        keys = [k for k in data.keys()]

        # _index_fn ensures that, between when we unpack
        # the sequence of samples in TrasformBatchType, and when we
        # pack them into a list of dictionaries, we always get the
        # same order of features. This is important because we don't
        # want one feature value accidentally getting mapped to the
        # wrong feature name
        def _index_fn(t: Tuple[str, Any]) -> int:
            k, _ = t
            return keys.index(k)

        to_transform_iterable = (
            dict(zip(keys, sample))
            for sample in zip(
                *(v for _, v in sorted(data.items(), key=_index_fn))
            )
        )
        transformed_batch: Dict[str, List[Any]] = {}
        for transformed_sample in self.transform(to_transform_iterable):
            for k, v in transformed_sample.items():
                transformed_batch.setdefault(k, []).append(v)

        return transformed_batch

    def get_dataset_fields(
        self: "HuggingFaceDatasetsInterfaceMapper", dataset: HfDatasetType
    ) -> Iterable[str]:
        return dataset.features.keys()

    def map(
        self: "HuggingFaceDatasetsInterfaceMapper",
        dataset: HfDatasetType,
        *_,
        **map_kwargs: Any
    ) -> HfDatasetType:

        self.check_dataset_fields(
            provided_fields=self.get_dataset_fields(dataset),
            expected_fields=self.input_fields,
        )

        if isinstance(self, BatchedBaseMapper):
            transformed_dataset = dataset.map(
                self._batch_transform, **{**map_kwargs, "batched": True}
            )
        elif isinstance(self, SingleBaseMapper):
            transformed_dataset = dataset.map(self.transform, **map_kwargs)
        else:
            raise TypeError(
                "Mapper must inherit a SingleBaseMapper or a BatchedBaseMapper"
            )

        self.check_dataset_fields(
            provided_fields=self.get_dataset_fields(transformed_dataset),
            expected_fields=self.output_fields,
        )

        return transformed_dataset


class TokensSequencesPaddingMapper(
    HuggingFaceDatasetsInterfaceMapper, multiseq.TokensSequencesPaddingMapper
):
    ...


class AttentionMaskSequencePaddingMapper(
    HuggingFaceDatasetsInterfaceMapper,
    multiseq.AttentionMaskSequencePaddingMapper,
):
    ...


class TokenTypeIdsSequencePaddingMapper(
    HuggingFaceDatasetsInterfaceMapper,
    multiseq.TokenTypeIdsSequencePaddingMapper,
):
    ...


class MakeAttentionMaskMapper(
    HuggingFaceDatasetsInterfaceMapper, multiseq.MakeAttentionMaskMapper
):
    ...


class LabelsMaskerMapper(
    HuggingFaceDatasetsInterfaceMapper, multiseq.LabelsMaskerMapper
):
    ...


class MultiSequenceStriderMapper(
    HuggingFaceDatasetsInterfaceMapper, multiseq.MultiSequenceStriderMapper
):
    ...


class SingleValueToSequenceMapper(
    HuggingFaceDatasetsInterfaceMapper, multiseq.SingleValueToSequenceMapper
):
    ...


class SequencesConcatenateMapper(
    HuggingFaceDatasetsInterfaceMapper, multiseq.SequencesConcatenateMapper
):
    ...


class FlattenMapper(HuggingFaceDatasetsInterfaceMapper, shape.FlattenMapper):
    ...


class BinarizerMapper(
    HuggingFaceDatasetsInterfaceMapper, shape.BinarizerMapper
):
    def map(
        self, dataset: HfDatasetType, *map_args: Any, **map_kwargs: Any
    ) -> HfDatasetType:
        dataset = super().map(dataset, *map_args, **map_kwargs)

        # we have to do this extra casting operation when dealing with
        # huggingface datasets because integer values are otherwise parsed
        # as floats.
        field_name, *_ = self.input_fields
        if isinstance(dataset.features[field_name], features.Sequence):
            new_field = features.Sequence(features.Value("int64"))
        else:
            new_field = features.Value("int64")
        dataset = dataset.cast_column(field_name, new_field)
        return dataset


class UnpackingMapper(
    HuggingFaceDatasetsInterfaceMapper, shape.UnpackingMapper
):
    ...


class TokenizerMapper(
    HuggingFaceDatasetsInterfaceMapper, tokenize.TokenizerMapper
):
    ...


class OneVsOtherAnnotatorMapper(
    HuggingFaceDatasetsInterfaceMapper, sse.OneVsOtherAnnotatorMapper
):
    ...


class ChangeFieldsMapper(
    HuggingFaceDatasetsInterfaceMapper, fields.ChangeFieldsMapper
):
    def map(
        self, dataset: HfDatasetType, *_, **map_kwargs: Any
    ) -> HfDatasetType:

        # mechanism to remove columns in huggingface datasets is
        # slightly different than in list of dict datasets.
        map_kwargs = {
            "remove_columns": list(dataset.features.keys()),
            **map_kwargs,
        }
        return super().map(dataset, **map_kwargs)


class ValidUnicodeMapper(
    HuggingFaceDatasetsInterfaceMapper, tokenize.ValidUnicodeMapper
):
    ...


class FixedBatchSizeMapper(
    HuggingFaceDatasetsInterfaceMapper, batchers.FixedBatchSizeMapper
):
    ...


class ListCollatorMapper(
    HuggingFaceDatasetsInterfaceMapper, collators.ListCollatorMapper
):
    ...


class FromTokenizerListCollatorMapper(
    HuggingFaceDatasetsInterfaceMapper,
    collators.FromTokenizerListCollatorMapper,
):
    ...


class TensorCollatorMapper(
    HuggingFaceDatasetsInterfaceMapper, collators.TensorCollatorMapper
):
    ...


class FromTokenizerTensorCollatorMapper(
    HuggingFaceDatasetsInterfaceMapper,
    collators.FromTokenizerTensorCollatorMapper,
):
    ...


class Python2TorchMapper(
    HuggingFaceDatasetsInterfaceMapper, converters.Python2TorchMapper
):
    def __init__(self: "Python2TorchMapper", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.device and self.device != torch.device("cpu"):
            raise RuntimeError(
                '"device" argument is not supported for Python2TorchMapper'
                " when using Huggingface datasets."
            )
        if len(self.field_cast_map) > 0:
            raise RuntimeError(
                '"field_cast_map" argument is not supported for '
                "Python2TorchMapper when using Huggingface datasets."
            )

    def map(self, dataset: HfDatasetType, *_, **__: Any) -> HfDatasetType:
        return dataset.with_format("torch")


class Torch2PythonMapper(
    HuggingFaceDatasetsInterfaceMapper, converters.Torch2PythonMapper
):
    def map(self, dataset: HfDatasetType, *_, **__: Any) -> HfDatasetType:
        # this changes the logic for converting to a python object
        # to map to apis for HuggingFace datasets
        return dataset.with_format(None)
