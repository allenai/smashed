from abc import ABCMeta, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from .dataset import BaseDataset
from .types import (
    Features,
    FeatureType,
    TransformBatchType,
    TransformElementType,
)

if TYPE_CHECKING:
    from .pipeline import Pipeline

__all__ = ["SingleBaseMapper", "BatchedBaseMapper"]


DatasetType = TypeVar("DatasetType", bound="BaseDataset")


class BaseMapper(metaclass=ABCMeta):
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
        self: "BaseMapper",
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

    @property
    @abstractmethod
    def batched(self: "BaseMapper") -> bool:
        raise NotImplementedError("batched property must be implemented")

    def _verify_output(self: "BaseMapper", data: Dict[str, Any]):
        for field in self.output_fields:
            if field not in data:
                raise ValueError(f"Field {field} not found in returned data")

    def __lshift__(
        self: "BaseMapper", other: Union["BaseMapper", "Pipeline"]
    ) -> "Pipeline":
        """Create a new Pipeline by combining this mapper with another."""

        # import actual implementation here to avoid circular imports
        from .pipeline import Pipeline

        return Pipeline(self) << other

    def __rshift__(self: "BaseMapper", other: "BaseMapper") -> "Pipeline":
        """Create a new Pipeline by combining this mapper with another."""
        return other << self

    def map(
        self: "BaseMapper", dataset: DatasetType, **map_kwargs: Any
    ) -> DatasetType:
        """Transform a dataset by applying this mapper's transform method.

        Args:
            dataset (DatasetType): The dataset to transform.
            **map_kwargs: Additional keyword arguments to pass to the
                transform method. By default, this is empty.

        Returns:
            dataset (DatasetType): The transformed dataset.
        """

        for field in self.input_fields:
            if field not in dataset.features:
                msg = f"Field {field} not found in input dataset {dataset}"
                raise ValueError(msg)

        map_kwargs = {"batched": self.batched, **map_kwargs}

        if self.batched:
            dataset = dataset.map(function=self._batch_transform, **map_kwargs)
        else:
            dataset = dataset.map(function=self.transform, **map_kwargs)

        # We optionally cast some (or all!) of the output columns
        # to a new format. By default, no casting occurs.
        to_cast_features = self.cast_columns(dataset.features)
        for feat_name, feat_type in to_cast_features.items():
            dataset = dataset.cast_column(feat_name, feat_type)

        for field in self.output_fields:
            if field not in dataset.features:
                raise ValueError(f"Field {field} not found in returned data")
        return dataset

    @abstractmethod
    def _batch_transform(
        self: "BaseMapper", data: TransformBatchType
    ) -> TransformBatchType:
        """Internal method for transforming a batch of data; this method
        is called by the map method when the mapper is batched.

        Any actual mapper implementation should NOT override this method;
        instead, any  mapper that operates on a batch should override the
        transform method, where the transform method receives a iterable
        of sample dictionaries as input.

        Args:
            data (TransformBatchType): The batch of data to transform.

        Returns:
            TransformBatchType: A batch of transformed data.
        """
        raise NotImplementedError(
            "Mapper subclass must implement _batch_transform"
        )

    @abstractmethod
    def transform(self: "BaseMapper", data: Any) -> Any:
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

    def cast_columns(
        self: "BaseMapper", features: Features
    ) -> Dict[str, FeatureType]:
        return {}


class SingleBaseMapper(BaseMapper, metaclass=ABCMeta):
    """An abstract implementation of a Mapper that operates on a single
    element. All mappers that operate on a single element should subclass
    this class.

    Actual mapper implementations should override the transform method.
    The transform method should accept a single sample dictionary as input
    and return a single sample dictionary as output.
    """

    @property
    def batched(self) -> bool:
        return False

    def _batch_transform(
        self: "BaseMapper", data: TransformBatchType
    ) -> TransformBatchType:
        raise ValueError("SingleBaseMapper does not support batching")

    @abstractmethod
    def transform(
        self: "BaseMapper", data: TransformElementType
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


class BatchedBaseMapper(BaseMapper, metaclass=ABCMeta):
    """An abstract implementation of a Mapper that operates on a batch of
    elements. All mappers that operate on a batch should subclass this
    class.

    Actual mapper implementations should override the transform method.
    The transform method should accept a iterator of dictionaries as input,
    and return a iterator of dictionaries as output. The number of samples
    returned may be different from the number of samples in the input.
    """

    @property
    def batched(self) -> bool:
        return True

    def _batch_transform(
        self: "BatchedBaseMapper", data: TransformBatchType
    ) -> TransformBatchType:
        """Internal method for transforming a batch of data; this method
        is called by the map method when the mapper is batched.

        Any actual mapper implementation should NOT override this method;
        instead, any  mapper that operates on a batch should override the
        transform method instead."""

        keys = [k for k in data.keys()]

        def _index_fn(t: Tuple[str, Any]) -> int:
            k, _ = t
            return keys.index(k)

        to_transform_iterable = (
            dict(zip(keys, sample))
            for sample in zip(
                *(v for _, v in sorted(data.items(), key=_index_fn))
            )
        )
        transformed_batch: Dict[str, List[Any]] = {}
        for transformed_sample in self.transform(to_transform_iterable):
            for k, v in transformed_sample.items():
                transformed_batch.setdefault(k, []).append(v)

        return transformed_batch

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
