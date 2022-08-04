from typing import Any, Dict, List, Mapping, Sequence

TransformElementType = Dict[str, Any]
TransformBatchType = Dict[str, List[Any]]
FeatureType = Any
Features = Mapping[str, FeatureType]
DatasetType = Sequence[TransformElementType]
