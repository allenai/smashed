from typing import Any, Callable, Iterable, Optional, Sequence, Type


from ..base import (BaseDataset, BaseMapper, Features, FeatureType,
                    TransformElementType)
from ..mappers import multiseq, shape, tokenize, fields
from ..mappers.contrib import sse


__all__ = [
    'Dataset',
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
    'MakeFieldMapper',
    'ValidUnicodeMapper'
]


class Dataset(list, BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def features(self) -> Features:
        if len(self):
            return {k: type(v) for k, v in self[0].items()}
        else:
            raise ValueError('Dataset is empty; cant get features')

    def keys(self) -> Sequence[str]:
        return [k for k in self.features.keys()]

    def map(
        self: 'Dataset',
        function: Optional[Callable] = None,
        batched: bool = False,
        *args,
        **kwargs
    ) -> 'Dataset':
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
        self: 'Dataset',
        column: str,
        feature: FeatureType,
        *_
    ) -> 'Dataset':

        def cast_fn(sample: TransformElementType) -> TransformElementType:
            return {k: (feature(v) if k == column else v)
                    for k, v in sample.items()}

        return self.map(cast_fn)


class _SimpleInterfaceMixInMapper(BaseMapper):
    def map(self, dataset: Dataset, **map_kwargs: Any) -> Dataset:
        return super().map(dataset, **map_kwargs)

    @classmethod
    def chain(cls: Type['_SimpleInterfaceMixInMapper'],
              dataset: Dataset,
              mappers: Sequence['_SimpleInterfaceMixInMapper'],
              **map_kwargs: Any) -> Dataset:
        return super().chain(dataset, mappers, **map_kwargs)

    def batch_transform(
        self: '_SimpleInterfaceMixInMapper',
        data: Iterable[TransformElementType]
    ) -> Iterable[TransformElementType]:
        # Simple mappers are called with slightly different data from
        # the base class, so we need to override this method.
        yield from super().transform(data)


class TokensSequencesPaddingMapper(
    _SimpleInterfaceMixInMapper,
    multiseq.TokensSequencesPaddingMapper
):
    ...


class AttentionMaskSequencePaddingMapper(
    _SimpleInterfaceMixInMapper,
    multiseq.AttentionMaskSequencePaddingMapper
):
    ...


class TokenTypeIdsSequencePaddingMapper(
    _SimpleInterfaceMixInMapper,
    multiseq.TokenTypeIdsSequencePaddingMapper
):
    ...


class MakeAttentionMaskMapper(
    _SimpleInterfaceMixInMapper,
    multiseq.MakeAttentionMaskMapper
):
    ...


class LabelsMaskerMapper(
    _SimpleInterfaceMixInMapper,
    multiseq.LabelsMaskerMapper
):
    ...


class MultiSequenceStriderMapper(
    _SimpleInterfaceMixInMapper,
    multiseq.MultiSequenceStriderMapper
):
    ...


class SingleValueToSequenceMapper(
    _SimpleInterfaceMixInMapper,
    multiseq.SingleValueToSequenceMapper
):
    ...


class SequencesConcatenateMapper(
    _SimpleInterfaceMixInMapper,
    multiseq.SequencesConcatenateMapper
):
    ...


class FlattenMapper(
    _SimpleInterfaceMixInMapper,
    shape.FlattenMapper
):
    ...


class BinarizerMapper(
    _SimpleInterfaceMixInMapper,
    shape.BinarizerMapper
):
    ...


class TokenizerMapper(
    _SimpleInterfaceMixInMapper,
    tokenize.TokenizerMapper
):
    ...


class OneVsOtherAnnotatorMapper(
    _SimpleInterfaceMixInMapper,
    sse.OneVsOtherAnnotatorMapper
):
    ...


class ChangeFieldsMapper(
    _SimpleInterfaceMixInMapper,
    fields.ChangeFieldsMapper
):
    ...


class MakeFieldMapper(
    _SimpleInterfaceMixInMapper,
    fields.MakeFieldMapper
):
    ...

class ValidUnicodeMapper(
    _SimpleInterfaceMixInMapper,
    tokenize.ValidUnicodeMapper
):
    ...