from smashed.base.dataset import BaseDataset
from ..utils import requires

requires("datasets")

from typing import Any, Protocol, Sequence, Type, TypeVar, Dict, overload

from datasets.arrow_dataset import Dataset
from datasets.iterable_dataset import IterableDataset
from datasets.features.features import Features, FeatureType


from ..base import BaseMapper
from ..mappers import multiseq, shape, tokenize, fields
from ..mappers.contrib import sse

from datasets import features


class HfDatasetProtocol(Dataset, BaseDataset):
    ...

class HfIterableDatasetProtocol(IterableDataset, BaseDataset):
    ...

HfDatasetType = TypeVar(
    "HfDatasetType", HfDatasetProtocol, HfIterableDatasetProtocol
)


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
    'ChangeFieldsMapper',
    'ValidUnicodeMapper'
]


class _HuggingFaceInterfaceMixInMapper(BaseMapper):
    def map(                            # type: ignore
        self,
        dataset: HfDatasetType,
        **map_kwargs: Any
    ) -> HfDatasetType:
        # this function is re-implemented to provide nice type annotations.
        # therefore, we remove the mypy type warnings with the ignore above.

        return super().map(dataset, **map_kwargs)

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


# We ignore type error bc they are due to the fact that the `map` method here
# maps to a BaseDataset vs the Dataset/IterableDataset type expected
# by _HuggingFaceInterfaceMixInMapper.
class ChangeFieldsMapper(                       # type: ignore
    _HuggingFaceInterfaceMixInMapper,
    fields.ChangeFieldsMapper
):
    ...


class ValidUnicodeMapper(
    _HuggingFaceInterfaceMixInMapper,
    tokenize.ValidUnicodeMapper
):
    ...
