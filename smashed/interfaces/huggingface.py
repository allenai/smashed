from abc import ABCMeta
from typing import Any, Dict, List, Tuple, TypeVar

from ..base.mapper import (
    AbstractBaseMapper,
    BatchedBaseMapper,
    SingleBaseMapper,
)
from ..base.types import TransformBatchType
from ..mappers import fields, multiseq, shape, tokenize
from ..mappers.contrib import sse
from ..utils import requires

requires("datasets", "2.4.0")

# we add the noqa bit because datasets is not part of the core requirement.
# Because we want to fail gracefully, `requires` above checks if `datasets`
# is installed, raising an helpful error message if it is not.
# However, that upsets the linter, which thinks that we should import in order
# instead. Therefore, we slap a bunch of `noqa`s in here to suppress it :)
from datasets import features  # noqa: E402
from datasets.arrow_dataset import Dataset  # noqa: E402
from datasets.iterable_dataset import IterableDataset  # noqa: E402

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
]


class HuggingFaceDatasetsInterfaceMapper(
    AbstractBaseMapper, metaclass=ABCMeta
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

    def map(self, dataset: HfDatasetType, **map_kwargs: Any) -> HfDatasetType:

        for field in self.input_fields:
            if field not in dataset.features:
                raise ValueError(f"Field {field} not found in dataset")

        if isinstance(self, BatchedBaseMapper):
            dataset = dataset.map(
                self._batch_transform, **{**map_kwargs, "batched": True}
            )
        elif isinstance(self, SingleBaseMapper):
            dataset = dataset.map(self.transform, **map_kwargs)
        else:
            raise TypeError(
                "Mapper must inherit a SingleBaseMapper or a BatchedBaseMapper"
            )

        for field in self.input_fields:
            if field not in dataset.features:
                raise ValueError(f"Field {field} not found in dataset")

        return dataset


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
    def map(self, dataset: HfDatasetType, **map_kwargs: Any) -> HfDatasetType:
        dataset = super().map(dataset, **map_kwargs)

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
    def map(self, dataset: HfDatasetType, **map_kwargs: Any) -> HfDatasetType:
        # columns need to be explicitly removed in huggingface datasets
        map_kwargs = {
            "remove_columns": list(dataset.features.keys()),
            **map_kwargs,
        }
        return super().map(dataset, **map_kwargs)


class ValidUnicodeMapper(
    HuggingFaceDatasetsInterfaceMapper, tokenize.ValidUnicodeMapper
):
    ...
