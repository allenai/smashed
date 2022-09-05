from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Iterable, List, Optional, Union

from ..types import TransformElementType
from .abstract import (
    AbstractBaseMapper,
    AbstractBatchedBaseMapper,
    AbstractSingleBaseMapper,
)
from .interfaces import MapMethodInterfaceMixIn

if TYPE_CHECKING:
    from ..pipeline import Pipeline


__all__ = ["SingleBaseMapper", "BatchedBaseMapper"]


class LshiftRshiftMixIn(AbstractBaseMapper):
    def __init__(
        self,
        input_fields: Optional[List[str]] = None,
        output_fields: Optional[List[str]] = None,
    ) -> None:
        """Create a new Mapper.

        Args:
            input_fields (Optional[List[str]], optional): The fields expected
                by this mapper. If None is provided, the mapper will not
                check for the presence of any input fields. Defaults to None.
            output_fields (Optional[List[str]], optional): The fields produced
                by this mapper after transformation. If None is provided, the
                mapper will not validate the output of transform. Defaults to
                None.
        """
        self.input_fields = input_fields or []
        self.output_fields = output_fields or []

    def __lshift__(
        self,
        other: Union["LshiftRshiftMixIn", "Pipeline"],
    ) -> "Pipeline":
        """Create a new Pipeline by combining this mapper with another."""
        # avoid circular import
        from ..pipeline import Pipeline

        return Pipeline(self) << other

    def __rshift__(
        self,
        other: Union["LshiftRshiftMixIn", "Pipeline"],
    ) -> "Pipeline":
        """Create a new Pipeline by combining this mapper with another."""
        return other << self


class SingleBaseMapper(
    MapMethodInterfaceMixIn,
    LshiftRshiftMixIn,
    AbstractSingleBaseMapper,
    metaclass=ABCMeta,
):
    """An abstract implementation of a Mapper that operates on a single
    element. All mappers that operate on a single element should subclass
    this class.

    Actual mapper implementations should override the transform method.
    The transform method should accept a single sample dictionary as input
    and return a single sample dictionary as output.
    """

    @abstractmethod
    def transform(self, data: TransformElementType) -> TransformElementType:
        """Transform a single sample of a dataset. This method should be
        overridden by actual mapper implementations.

        Args:
            data (TransformElementType): The sample to transform. This is
                a single sample dictionary with string keys and values of
                any type.

        Returns:
            TransformElementType: The transformed sample. This is a single
                sample dictionary with string keys and values of any type.
                The keys can be different from the input keys.
        """
        raise NotImplementedError("Mapper subclass must implement transform")


class BatchedBaseMapper(
    MapMethodInterfaceMixIn,
    LshiftRshiftMixIn,
    AbstractBatchedBaseMapper,
    metaclass=ABCMeta,
):
    """An abstract implementation of a Mapper that operates on a batch of
    elements. All mappers that operate on a batch should subclass this
    class.

    Actual mapper implementations should override the transform method.
    The transform method should accept a iterator of dictionaries as input,
    and return a iterator of dictionaries as output. The number of samples
    returned may be different from the number of samples in the input.
    """

    @abstractmethod
    def transform(
        self, data: Iterable[TransformElementType]
    ) -> Iterable[TransformElementType]:
        """Transform a batch of data. This method should be overridden by
        actual mapper implementations.

        Args:
            data (Iterable[TransformElementType]): The batch of data to
                transform; it is an iterable of dictionaries, where each
                dictionary is a sample (i.e. a Dict[str, Any]).

        Returns:
            Iterable[TransformElementType]: Iterable of transformed data.
                Each element in the iterable is a dictionary with str keys
                and values of any type. The number of elements in the
                iterable may be different from the number of samples in the
                input.
        """
        raise NotImplementedError("Mapper subclass must implement transform")
