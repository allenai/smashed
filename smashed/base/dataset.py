from typing import Callable, Optional, Protocol, TypeVar

from .types import FeatureType, Features


D = TypeVar('D', bound='BaseDataset')


class BaseDataset(Protocol):
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
