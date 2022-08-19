from typing import Any, Dict, List, Mapping, Sequence, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from torchdata.datapipes.map import SequenceWrapper
    from torchdata.datapipes.iter import IterableWrapper

    from datasets.arrow_dataset import Dataset
    from datasets.iterable_dataset import IterableDataset


TransformElementType = Dict[str, Any]
TransformBatchType = Dict[str, List[Any]]
FeatureType = Any
Features = Mapping[str, FeatureType]

# Representations of datasets
DatasetType = Sequence[TransformElementType]
TorchDataDatasetType = Union[SequenceWrapper, IterableWrapper]
HuggingFaceDatasetType = Union[Dataset, IterableDataset]
