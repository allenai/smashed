from typing import Any, Generic, Iterable, Tuple, TypeVar, Union

from .types import TransformElementType

D = TypeVar("D")
S = TypeVar("S")


class AbstractBaseMapper(Generic[D, S]):
    """An abstract implementation of a Mapper"""

    input_fields: Tuple[str, ...]
    output_fields: Tuple[str, ...]
    fingerprint: str
    pipeline: Union["AbstractBaseMapper", None]

    def __init__(self, *args, **kwargs):
        ...

    def map(self, dataset: D, **map_kwargs: Any) -> D:
        """Transform a dataset by applying this mapper's transform method.

        Args:
            dataset (DatasetType): The dataset to transform.
            **map_kwargs: Additional keyword arguments to pass to the
                transform method. By default, this is empty.

        Returns:
            dataset (DatasetType): The transformed dataset.
        """
        raise NotImplementedError(
            f"Subclasses {self.__class__.__name__} must implement map"
        )

    def transform(self, data: S) -> S:
        """Apply the transformation for this mapper. This method should be
        overridden by actual mapper implementations.

        Args:
            data (TransformElementType): The sample to transform. This is
                a single sample dictionary with string keys and values of
                any type, or an iterable of such dictionaries.

        Returns:
            TransformElementType: The transformed sample. This is a single
                sample dictionary with string keys and values of any type,
                or an iterable of such dictionaries. It should have the
                same type as the input.

        """
        raise NotImplementedError(
            f"Subclasses {self.__class__.__name__} must implement transform"
        )


class AbstractSingleBaseMapper(AbstractBaseMapper):
    """An abstract implementation of a Mapper that operates on a single
    element."""

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
        raise NotImplementedError(
            f"Subclasses {self.__class__.__name__} must implement transform"
        )


class AbstractBatchedBaseMapper(AbstractBaseMapper):
    """An abstract implementation of a Mapper that operates on a batch of
    elements."""

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
        raise NotImplementedError(
            f"Subclasses {self.__class__.__name__} must implement transform"
        )
