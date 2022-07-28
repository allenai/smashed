from typing import Any, Dict, TypeVar

from smashed.base.dataset import BaseDataset

from ..base import BaseMapper
from ..mappers import fields, multiseq, shape, tokenize
from ..mappers.contrib import sse
from ..utils import requires

requires("datasets")

# we add the noqa bit because datasets is not part of the core requirement.
# Because we want to fail gracefully, `requires` above checks if `datasets`
# is installed, raising an helpful error message if it is not.
# However, that upsets the linter, which thinks that we should import in order
# instead. Therefore, we slap a bunch of `noqa`s in here to suppress it :)
from datasets import features  # noqa: E402
from datasets.arrow_dataset import Dataset  # noqa: E402
from datasets.features.features import Features, FeatureType  # noqa: E402
from datasets.iterable_dataset import IterableDataset  # noqa: E402


class HfDatasetProtocol(Dataset, BaseDataset):
    ...


class HfIterableDatasetProtocol(IterableDataset, BaseDataset):
    ...


HfDatasetType = TypeVar(
    "HfDatasetType", HfDatasetProtocol, HfIterableDatasetProtocol
)


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
]


class _HuggingFaceInterfaceMixInMapper(BaseMapper):
    def map(  # type: ignore
        self, dataset: HfDatasetType, **map_kwargs: Any
    ) -> HfDatasetType:
        # this function is re-implemented to provide nice type annotations.
        # therefore, we remove the mypy type warnings with the ignore above.

        return super().map(dataset, **map_kwargs)

    def cast_columns(self, features: Features) -> Dict[str, FeatureType]:
        return super().cast_columns(features)


class TokensSequencesPaddingMapper(
    _HuggingFaceInterfaceMixInMapper, multiseq.TokensSequencesPaddingMapper
):
    ...


class AttentionMaskSequencePaddingMapper(
    _HuggingFaceInterfaceMixInMapper,
    multiseq.AttentionMaskSequencePaddingMapper,
):
    ...


class TokenTypeIdsSequencePaddingMapper(
    _HuggingFaceInterfaceMixInMapper,
    multiseq.TokenTypeIdsSequencePaddingMapper,
):
    ...


class MakeAttentionMaskMapper(
    _HuggingFaceInterfaceMixInMapper, multiseq.MakeAttentionMaskMapper
):
    ...


class LabelsMaskerMapper(
    _HuggingFaceInterfaceMixInMapper, multiseq.LabelsMaskerMapper
):
    ...


class MultiSequenceStriderMapper(
    _HuggingFaceInterfaceMixInMapper, multiseq.MultiSequenceStriderMapper
):
    ...


class SingleValueToSequenceMapper(
    _HuggingFaceInterfaceMixInMapper, multiseq.SingleValueToSequenceMapper
):
    ...


class SequencesConcatenateMapper(
    _HuggingFaceInterfaceMixInMapper, multiseq.SequencesConcatenateMapper
):
    ...


class FlattenMapper(_HuggingFaceInterfaceMixInMapper, shape.FlattenMapper):
    ...


class BinarizerMapper(_HuggingFaceInterfaceMixInMapper, shape.BinarizerMapper):
    __value_type__: type = features.Value
    __sequence_type__: type = features.Sequence


class TokenizerMapper(
    _HuggingFaceInterfaceMixInMapper, tokenize.TokenizerMapper
):
    ...


class OneVsOtherAnnotatorMapper(
    _HuggingFaceInterfaceMixInMapper, sse.OneVsOtherAnnotatorMapper
):
    ...


# We ignore type error bc they are due to the fact that the `map` method here
# maps to a BaseDataset vs the Dataset/IterableDataset type expected
# by _HuggingFaceInterfaceMixInMapper.
class ChangeFieldsMapper(  # type: ignore
    _HuggingFaceInterfaceMixInMapper, fields.ChangeFieldsMapper
):
    ...


class ValidUnicodeMapper(
    _HuggingFaceInterfaceMixInMapper, tokenize.ValidUnicodeMapper
):
    ...
