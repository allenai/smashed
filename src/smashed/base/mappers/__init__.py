import inspect
from typing import Union

from .base import BatchedBaseMapper, SingleBaseMapper

__all__ = ["SingleBaseMapper", "BatchedBaseMapper"]


def is_mapper_cls(cls_: object) -> bool:
    """Check if a class is a mapper."""
    return inspect.isclass(cls_) and issubclass(
        cls_, (SingleBaseMapper, BatchedBaseMapper)
    )


def is_mapper_obj(obj: object) -> bool:
    """Check if an object is a mapper."""
    return isinstance(obj, (SingleBaseMapper, BatchedBaseMapper))


def is_mapper(cls_or_obj: Union[type, object]) -> bool:
    """Check if a class or object is a mapper."""
    return is_mapper_cls(cls_or_obj) or is_mapper_obj(cls_or_obj)
