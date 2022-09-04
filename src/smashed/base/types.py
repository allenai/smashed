from typing import Any, Dict, Iterable, List, Mapping, Sequence, TypeVar, Union

from necessary import necessary

if necessary("torchdata", soft=True):
    from torchdata.datapipes.iter import (
        IterableWrapper as TorchDataIterableDataset
    )
    from torchdata.datapipes.map import (
        SequenceWrapper as TorchDataSequenceDataset
    )
else:
    TorchDataSequenceDataset = Sequence
    TorchDataIterableDataset = Iterable

with necessary("datasets", soft=True) as HUGGINGFACE_DATASET_AVAILABLE:
    if HUGGINGFACE_DATASET_AVAILABLE:
        from datasets.arrow_dataset import (
            Dataset as HuggingFaceDataset
        )
        from datasets.iterable_dataset import (
            IterableDataset as HuggingFaceIterableDataset
        )
    else:
        HuggingFaceDataset = Sequence
        HuggingFaceIterableDataset = Iterable

TransformElementType = Dict[str, Any]
TransformBatchType = Dict[str, List[Any]]
FeatureType = Any
Features = Mapping[str, FeatureType]

# Representations of datasets
ListOfDictsDatasetType = Sequence[TransformElementType]
TorchDataDatasetType = Union[
    TorchDataSequenceDataset, TorchDataIterableDataset
]
HuggingFaceDatasetType = Union[
    HuggingFaceDataset, HuggingFaceIterableDataset
]

print(reveal_locals())
