from typing import Any, Dict, Iterable, List, Mapping, Sequence, Union

from necessary import necessary

if necessary("torchdata", soft=True):
    from torchdata.datapipes.iter import IterableWrapper
    from torchdata.datapipes.map import SequenceWrapper
else:
    SequenceWrapper = Sequence
    IterableWrapper = Iterable

if necessary("datasets", soft=True):
    from datasets.arrow_dataset import Dataset
    from datasets.iterable_dataset import IterableDataset
else:
    Dataset = Sequence
    IterableDataset = Iterable


TransformElementType = Dict[str, Any]
TransformBatchType = Dict[str, List[Any]]
FeatureType = Any
Features = Mapping[str, FeatureType]

# Representations of datasets
DatasetType = Sequence[TransformElementType]
TorchDataDatasetType = Union[SequenceWrapper, IterableWrapper]
HuggingFaceDatasetType = Union[Dataset, IterableDataset]
