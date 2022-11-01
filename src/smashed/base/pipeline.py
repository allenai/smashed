from typing import TypeVar, Union

from .mappers import BatchedBaseMapper, SingleBaseMapper

M = TypeVar("M", "SingleBaseMapper", "BatchedBaseMapper")


def make_pipeline(
    first_mapper: M,
    *rest_mappers: Union["SingleBaseMapper", "BatchedBaseMapper"]
) -> M:
    """Make a pipeline of mappers."""
    for mapper in rest_mappers:
        first_mapper = first_mapper.chain(mapper)
    return first_mapper
