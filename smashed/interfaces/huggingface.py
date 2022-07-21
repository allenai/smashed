from ..utils import requires

requires("datasets")

from typing import Any, Sequence, Type, TypeVar, Dict

from datasets.arrow_dataset import Dataset
from datasets.iterable_dataset import IterableDataset
from datasets.features.features import Features, FeatureType


from ..base import BaseMapper
from ..mappers import multiseq, shape, tokenize, fields
from ..mappers.contrib import sse

from datasets import features


HfDatasetType = TypeVar("HfDatasetType", Dataset, IterableDataset)


__all__ = [
    'TokensSequencesPaddingMapper',
    'AttentionMaskSequencePaddingMapper',
    'TokenTypeIdsSequencePaddingMapper',
    'MakeAttentionMaskMapper',
    'LabelsMaskerMapper',
    'MultiSequenceStriderMapper',
    'SingleValueToSequenceMapper',
    'SequencesConcatenateMapper',
    'FlattenMapper',
    'BinarizerMapper',
    'TokenizerMapper',
    'OneVsOtherAnnotatorMapper',
    'ChangeFieldsMapper'
]


class _HuggingFaceInterfaceMixInMapper(BaseMapper):

    def map(self, dataset: Dataset, **map_kwargs: Any) -> Dataset:
        return super().map(dataset, **map_kwargs)

    @classmethod
    def chain(cls: Type['_HuggingFaceInterfaceMixInMapper'],
              dataset: HfDatasetType,
              mappers: Sequence['_HuggingFaceInterfaceMixInMapper'],
              **map_kwargs: Any) -> HfDatasetType:
        ...

    def cast_columns(self, features: Features) -> Dict[str, FeatureType]:
        return super().cast_columns(features)


class TokensSequencesPaddingMapper(
    _HuggingFaceInterfaceMixInMapper,
    multiseq.TokensSequencesPaddingMapper
):
    ...


class AttentionMaskSequencePaddingMapper(
    _HuggingFaceInterfaceMixInMapper,
    multiseq.AttentionMaskSequencePaddingMapper
):
    ...


class TokenTypeIdsSequencePaddingMapper(
    _HuggingFaceInterfaceMixInMapper,
    multiseq.TokenTypeIdsSequencePaddingMapper
):
    ...


class MakeAttentionMaskMapper(
    _HuggingFaceInterfaceMixInMapper,
    multiseq.MakeAttentionMaskMapper
):
    ...


class LabelsMaskerMapper(
    _HuggingFaceInterfaceMixInMapper,
    multiseq.LabelsMaskerMapper
):
    ...


class MultiSequenceStriderMapper(
    _HuggingFaceInterfaceMixInMapper,
    multiseq.MultiSequenceStriderMapper
):
    ...


class SingleValueToSequenceMapper(
    _HuggingFaceInterfaceMixInMapper,
    multiseq.SingleValueToSequenceMapper
):
    ...


class SequencesConcatenateMapper(
    _HuggingFaceInterfaceMixInMapper,
    multiseq.SequencesConcatenateMapper
):
    ...


class FlattenMapper(
    _HuggingFaceInterfaceMixInMapper,
    shape.FlattenMapper
):
    ...


class BinarizerMapper(
    _HuggingFaceInterfaceMixInMapper,
    shape.BinarizerMapper
):
    __value_type__: type = features.Value
    __sequence_type__: type = features.Sequence


class TokenizerMapper(
    _HuggingFaceInterfaceMixInMapper,
    tokenize.TokenizerMapper
):
    ...


class OneVsOtherAnnotatorMapper(
    _HuggingFaceInterfaceMixInMapper,
    sse.OneVsOtherAnnotatorMapper
):
    ...


class ChangeFieldsMapper(
    _HuggingFaceInterfaceMixInMapper,
    fields.ChangeFieldsMapper
):
    ...
