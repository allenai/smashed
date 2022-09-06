import hashlib
import inspect
import pickle
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Iterable, List, NamedTuple, Optional, Union

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


class PipelineFingerprintMixIn(AbstractBaseMapper):
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
        self.fingerprint = self._get_mapper_fingerprint()

    def __lshift__(
        self,
        other: Union["PipelineFingerprintMixIn", "Pipeline"],
    ) -> "Pipeline":
        """Create a new Pipeline by combining this mapper with another."""
        # avoid circular import
        from ..pipeline import Pipeline

        return Pipeline(self) << other

    def __rshift__(
        self,
        other: Union["PipelineFingerprintMixIn", "Pipeline"],
    ) -> "Pipeline":
        """Create a new Pipeline by combining this mapper with another."""
        return other << self

    def _get_mapper_fingerprint(self) -> str:
        """Compute a hash for this mapper; the hash depends of the arguments
        passed to the constructor, NOT of the data passed to the mapper or
        the state of the mapper."""

        if hasattr(self, "fingerprint"):
            # don't recompute the fingerprint if it's already been computed
            return self.fingerprint

        class ExtInfo(NamedTuple):
            frame_info: inspect.FrameInfo
            arg_info: inspect.ArgInfo

        stack_frames = [
            ExtInfo(arg_info=inspect.getargvalues(fr.frame), frame_info=fr)
            for fr in inspect.stack()
        ]

        init_calls = [
            frame
            for frame in stack_frames
            if (
                # To be a frame associated with the init call, it must...
                #   1. ...be a method call, i.e. have a 'self' argument
                "self" in frame.arg_info.args
                #   2. ...be an instance or subclass of this class
                and isinstance(
                    frame.arg_info.locals["self"], AbstractBaseMapper
                )
                #   3. ...be the __init__ method
                and frame.frame_info.function == "__init__"
            )
        ]

        def _get_cls_name_from_frame_info(frame_ext_info: ExtInfo) -> str:
            cls_ = frame_ext_info.arg_info.locals.get(
                "__class__", PipelineFingerprintMixIn
            )
            return f"{cls_.__module__}.{cls_.__name__}"

        signature = {
            _get_cls_name_from_frame_info(frame): {
                arg: frame.arg_info.locals[arg]
                for arg in frame.arg_info.args
                if arg != "self"
            }
            for frame in init_calls
        }

        # get sha1 hash of the signature
        (sha1 := hashlib.sha1()).update(pickle.dumps(signature))
        return sha1.hexdigest()


class SingleBaseMapper(
    MapMethodInterfaceMixIn,
    PipelineFingerprintMixIn,
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
    PipelineFingerprintMixIn,
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
