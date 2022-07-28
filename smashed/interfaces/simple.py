from typing import Any, Callable, Optional, Sequence

from ..base.dataset import BaseDataset
from ..base.mapper import BaseMapper
from ..base.types import (
    Features,
    FeatureType,
    TransformBatchType,
    TransformElementType,
)
from ..mappers import fields, multiseq, shape, tokenize
from ..mappers.contrib import sse

__all__ = [
    "Dataset",
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
    "MakeFieldMapper",
    "ValidUnicodeMapper",
]


class Dataset(list, BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def features(self) -> Features:
        if len(self):
            return {k: type(v) for k, v in self[0].items()}
        else:
            raise ValueError("Dataset is empty; cant get features")

    def keys(self) -> Sequence[str]:
        return [k for k in self.features.keys()]

    def map(
        self: "Dataset",
        function: Optional[Callable] = None,
        batched: bool = False,
        *args,
        **kwargs,
    ) -> "Dataset":
        if function is None:
            return self

        processed = Dataset()
        for sample in self:
            if batched:
                processed.extend(function([sample]))
            else:
                processed.append(function(sample))

        return processed

    def cast_column(
        self: "Dataset", column: str, feature: FeatureType, *_
    ) -> "Dataset":
        def cast_fn(sample: TransformElementType) -> TransformElementType:
            return {
                k: (feature(v) if k == column else v)
                for k, v in sample.items()
            }

        return self.map(cast_fn)


class _SimpleInterfaceMixInMapper(BaseMapper):
    def map(  # type: ignore
        self: "_SimpleInterfaceMixInMapper",
        dataset: Dataset,
        **map_kwargs: Any,
    ) -> Dataset:
        # this function is re-implemented to provide nice
        # type annotations.

        if not isinstance(dataset, Dataset):
            raise ValueError(
                f"Dataset must be of type `{Dataset}`"
                f" but I got `{type(dataset)}` instead."
            )

        # This ensures that the super for the next in line in the mro,
        # which should be whatever actual implementation of the mapper
        # this mixin is mixed in with, will be called.
        return super().map(dataset, **map_kwargs)

    @property
    def batched(self: "_SimpleInterfaceMixInMapper") -> bool:
        # This ensures that the super for the next in line in the mro,
        # which should be whatever actual implementation of the mapper
        # this mixin is mixed in with, will be called.
        return super().batched

    def _batch_transform(  # type: ignore
        self,  # type: ignore
        data: TransformBatchType,  # type: ignore
    ) -> TransformBatchType:  # type: ignore
        # Simple mappers are called with slightly different data from
        # the base class, so we need to override this method.

        # Note that we need to disable the type check to avoid a mess
        # with mypy. It's a bit of a hack, because the type returned here is
        # not what mypy expects, but it works as long as Dataset is an
        # instance of a list.

        yield from super().transform(data)


class TokensSequencesPaddingMapper(
    _SimpleInterfaceMixInMapper, multiseq.TokensSequencesPaddingMapper
):
    ...


class AttentionMaskSequencePaddingMapper(
    _SimpleInterfaceMixInMapper, multiseq.AttentionMaskSequencePaddingMapper
):
    ...


class TokenTypeIdsSequencePaddingMapper(
    _SimpleInterfaceMixInMapper, multiseq.TokenTypeIdsSequencePaddingMapper
):
    ...


class MakeAttentionMaskMapper(
    _SimpleInterfaceMixInMapper, multiseq.MakeAttentionMaskMapper
):
    ...


class LabelsMaskerMapper(
    _SimpleInterfaceMixInMapper, multiseq.LabelsMaskerMapper
):
    ...


class MultiSequenceStriderMapper(
    _SimpleInterfaceMixInMapper, multiseq.MultiSequenceStriderMapper
):
    ...


class SingleValueToSequenceMapper(
    _SimpleInterfaceMixInMapper, multiseq.SingleValueToSequenceMapper
):
    ...


class SequencesConcatenateMapper(
    _SimpleInterfaceMixInMapper, multiseq.SequencesConcatenateMapper
):
    ...


class FlattenMapper(_SimpleInterfaceMixInMapper, shape.FlattenMapper):
    ...


class BinarizerMapper(_SimpleInterfaceMixInMapper, shape.BinarizerMapper):
    ...


class TokenizerMapper(_SimpleInterfaceMixInMapper, tokenize.TokenizerMapper):
    ...


class OneVsOtherAnnotatorMapper(
    _SimpleInterfaceMixInMapper, sse.OneVsOtherAnnotatorMapper
):
    ...


# We ignore type error bc they are due to the fact that
# the `map` method here maps to a BaseDataset vs the Dataset
# type expected by _SimpleInterfaceMixInMapper.
class ChangeFieldsMapper(  # type: ignore
    _SimpleInterfaceMixInMapper, fields.ChangeFieldsMapper
):
    ...


class MakeFieldMapper(_SimpleInterfaceMixInMapper, fields.MakeFieldMapper):
    ...


class ValidUnicodeMapper(
    _SimpleInterfaceMixInMapper, tokenize.ValidUnicodeMapper
):
    ...
