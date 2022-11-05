from .abstract import AbstractBaseMapper
from .mappers import BatchedBaseMapper, SingleBaseMapper
from .pipeline import make_pipeline
from .recipes import BaseRecipe
from .types import TransformBatchType, TransformElementType
from .views import DataBatchView, DataRowView

# create a shortcut for the base class; this is
# useful for cases where you want to check if object is
# a mapper or indicate that a mapper of any type is returned
BaseMapper = AbstractBaseMapper


__all__ = [
    "BaseMapper",
    "BaseRecipe",
    "BatchedBaseMapper",
    "DataBatchView",
    "DataRowView",
    "make_pipeline",
    "SingleBaseMapper",
    "TransformBatchType",
    "TransformElementType",
]
