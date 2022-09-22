from typing import Iterable

from ..base import BatchedBaseMapper, SingleBaseMapper, TransformElementType


class DebugSingleMapper(SingleBaseMapper):
    def transform(self, data: TransformElementType) -> TransformElementType:
        breakpoint()
        return data


class DebugBatchedMapper(BatchedBaseMapper):
    def transform(
        self, data: Iterable[TransformElementType]
    ) -> Iterable[TransformElementType]:
        breakpoint()
        return data
