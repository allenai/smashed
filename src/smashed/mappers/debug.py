import os
from typing import Any, Iterable, Optional, TypeVar

from ..base import BatchedBaseMapper, SingleBaseMapper, TransformElementType
from ..base.abstract import AbstractBaseMapper


class DebugSingleMapper(SingleBaseMapper):
    """A single mapper that inserts a breakpoint into the transform"""

    def __init__(self, use_ipdb: bool = False):
        if use_ipdb:
            os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"

    def transform(self, data: TransformElementType) -> TransformElementType:
        breakpoint()
        return data


class DebugBatchedMapper(BatchedBaseMapper):
    """A batched mapper that inserts a breakpoint into the transform"""

    def __init__(self, use_ipdb: bool = False):
        if use_ipdb:
            os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"

    def transform(
        self, data: Iterable[TransformElementType]
    ) -> Iterable[TransformElementType]:
        breakpoint()
        return data


V = TypeVar("V", int, float, str, bool, list, tuple)


class _MockMapperMixIn(AbstractBaseMapper):
    def __init__(
        self,
        value: Any,
        input_fields: Optional[Iterable[str]] = None,
        output_fields: Optional[Iterable[str]] = None,
    ):
        super().__init__(
            input_fields=input_fields,
            output_fields=output_fields,
        )
        self.value = value
        self.default = type(value)


class MockMapper(_MockMapperMixIn, SingleBaseMapper):
    """A single mapper that returns the same data it receives.
    Used for testing."""

    def transform(self, data: TransformElementType) -> TransformElementType:
        return {k: v + self.value for k, v in data.items()}


class BatchMockMapper(_MockMapperMixIn, BatchedBaseMapper):
    """A batched mapper that returns the same data it receives.
    Used for testing."""

    def transform(
        self, data: Iterable[TransformElementType]
    ) -> Iterable[TransformElementType]:
        for d in data:
            yield {k: v + self.value for k, v in d.items()}
