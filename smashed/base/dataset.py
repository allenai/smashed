from typing import Callable, Optional, Protocol, TypeVar

from .types import FeatureType, Features


D = TypeVar('D', bound='BaseDataset')


class BaseDataset(Protocol):
    """A protocol for datasets. In general, a dataset is a collection of samples
    that can be iterated over. Datasets support a map operation that can either
    transform a sample into another sample (1-to-1 mapping) or generate
    multiple samples from a single sample (1-to-many mapping).

    Dataset APIs are modeled after HuggingFace ArrowDataset APIs."""


    @property
    def features(self) -> Features:
        ...

    def map(self: D,
            function: Optional[Callable] = None,
            *,
            batched: bool = False,
            batch_size: Optional[int] = 1000) -> D:
        ...

    def cast_column(
        self: D,
        column: str,
        feature: FeatureType,
        *_
    ) -> D:
        ...
