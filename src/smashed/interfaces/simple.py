from ..mappers.batchers import FixedBatchSizeMapper
from ..mappers.collators import (
    FromTokenizerListCollatorMapper,
    FromTokenizerTensorCollatorMapper,
    ListCollatorMapper,
    TensorCollatorMapper,
)
from ..mappers.contrib.sse import OneVsOtherAnnotatorMapper
from ..mappers.converters import Python2TorchMapper, Torch2PythonMapper
from ..mappers.fields import ChangeFieldsMapper, MakeFieldMapper
from ..mappers.multiseq import (
    AttentionMaskSequencePaddingMapper,
    LabelsMaskerMapper,
    MakeAttentionMaskMapper,
    MultiSequenceStriderMapper,
    SequencesConcatenateMapper,
    SingleValueToSequenceMapper,
    TokensSequencesPaddingMapper,
    TokenTypeIdsSequencePaddingMapper,
)
from ..mappers.shape import BinarizerMapper, FlattenMapper, UnpackingMapper
from ..mappers.tokenize import TokenizerMapper, ValidUnicodeMapper
from ..utils import SmashedWarnings

__all__ = [
    "Dataset",
    "TokensSequencesPaddingMapper",
    "AttentionMaskSequencePaddingMapper",
    "TokenTypeIdsSequencePaddingMapper",
    "UnpackingMapper",
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
    "FixedBatchSizeMapper",
    "ListCollatorMapper",
    "TensorCollatorMapper",
    "FromTokenizerListCollatorMapper",
    "FromTokenizerTensorCollatorMapper",
    "Python2TorchMapper",
    "Torch2PythonMapper",
]


class Dataset(list):
    def __new__(cls, *args, **kwargs) -> list:  # type: ignore
        SmashedWarnings.deprecation(
            "smashed.interfaces.simple.Dataset is deprecated; "
            "simply use a list of dictionaries with str keys instead."
        )
        return list(*args, **kwargs)
