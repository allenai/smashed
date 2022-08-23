from typing import Any, Dict, List, Mapping, Sequence, Iterable, Union

from ..utils import requires

if requires("torchdata", soft=True):
    from torchdata.datapipes.map import SequenceWrapper
    from torchdata.datapipes.iter import IterableWrapper
else:
    SequenceWrapper = Sequence
    IterableWrapper = Iterable

if requires("datasets", soft=True):
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
