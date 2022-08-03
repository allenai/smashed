from abc import ABCMeta, abstractmethod
from collections import abc
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterable,
    List,
    Optional,
    TypeVar,
    Union,
)

from .types import DatasetType, TransformElementType

if TYPE_CHECKING:
    from .pipeline import Pipeline

__all__ = ["SingleBaseMapper", "BatchedBaseMapper"]


D = TypeVar("D")
S = TypeVar("S")


class AbstractBaseMapper(Generic[D, S], metaclass=ABCMeta):
    """An abstract implementation of a Mapper. Do not subclass directly;
    instead subclass either SimpleBaseMapper (if your mapper operates on
    a single element) or BatchedBaseMapper (if your mapper operates on a
    batch).

    A mapper is an object that transforms samples in a dataset. All mappers
    do so by implementing a method called `transform`.

    Optionally, mappers can also implement the cast_columns method,
    which returns a dictionary of new types for some (or all) of the
    input/output columns.
    """

    __slots__: List[str] = ["input_fields", "output_fields"]
    input_fields: List[str]
    output_fields: List[str]

    def __init__(
        self: "AbstractBaseMapper",
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
        self: "AbstractBaseMapper",
        other: Union["AbstractBaseMapper", "Pipeline"],
    ) -> "Pipeline":
        """Create a new Pipeline by combining this mapper with another."""

        # import actual implementation here to avoid circular imports
        from .pipeline import Pipeline

        return Pipeline(self) << other

    def __rshift__(
        self: "AbstractBaseMapper",
        other: Union["AbstractBaseMapper", "Pipeline"],
    ) -> "Pipeline":
        """Create a new Pipeline by combining this mapper with another."""
        return other << self

    @abstractmethod
    def map(self: "AbstractBaseMapper", dataset: D, **map_kwargs: Any) -> D:
        """Transform a dataset by applying this mapper's transform method.

        Args:
            dataset (DatasetType): The dataset to transform.
            **map_kwargs: Additional keyword arguments to pass to the
                transform method. By default, this is empty.

        Returns:
            dataset (DatasetType): The transformed dataset.
        """
        raise NotImplementedError("map method must be implemented")

    @abstractmethod
    def transform(self: "AbstractBaseMapper", data: S) -> S:
        """Transform a single sample of a dataset. This method should be
        overridden by actual mapper implementations.

        Args:
            self (BaseMapper): _description_
            data (Any): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            Any: _description_
        """
        raise NotImplementedError("Mapper subclass must implement transform")


class ListOfDictsDatasetInterfaceMapper(AbstractBaseMapper, metaclass=ABCMeta):
    """A mixin class for a mapper that operates on a list of dictionaries.
    It's the default mapper type.
    """

    def map(
        self: "ListOfDictsDatasetInterfaceMapper",
        dataset: DatasetType,
        **_: Any,
    ) -> DatasetType:
        if isinstance(dataset, abc.Sequence):
            for field in self.input_fields:
                if field not in dataset[0]:
                    raise ValueError(f"Field {field} not found in dataset")

        if isinstance(self, BatchedBaseMapper):
            transformed_dataset = list(self.transform(dataset))
        elif isinstance(self, SingleBaseMapper):
            transformed_dataset = [
                self.transform(sample) for sample in dataset
            ]
        else:
            raise TypeError(
                "Mapper must inherit a SingleBaseMapper or a BatchedBaseMapper"
            )

        for field in self.output_fields:
            if field not in transformed_dataset[0]:
                raise ValueError(f"Field {field} not found in dataset")

        return transformed_dataset


class SingleBaseMapper(ListOfDictsDatasetInterfaceMapper, metaclass=ABCMeta):
    """An abstract implementation of a Mapper that operates on a single
    element. All mappers that operate on a single element should subclass
    this class.

    Actual mapper implementations should override the transform method.
    The transform method should accept a single sample dictionary as input
    and return a single sample dictionary as output.
    """

    @abstractmethod
    def transform(
        self: "AbstractBaseMapper", data: TransformElementType
    ) -> TransformElementType:
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


class BatchedBaseMapper(ListOfDictsDatasetInterfaceMapper, metaclass=ABCMeta):
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
        self: "BatchedBaseMapper", data: Iterable[TransformElementType]
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
