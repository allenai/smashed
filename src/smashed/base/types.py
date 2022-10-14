from typing import Any, Dict, List, Mapping, Union

from .views import DataBatchView, DataRowView

TransformElementType = Union[Dict[str, Any], DataRowView[str, Any]]
TransformBatchType = Union[
    Dict[str, List[Any]], DataBatchView[Any, str, List[Any]]
]
FeatureType = Any
Features = Mapping[str, FeatureType]
