import copy
import hashlib
import inspect
import pickle
from itertools import chain
from typing import Iterable, List, NamedTuple, Optional, TypeVar, Union

from ..utils import bytes_from_int, int_from_bytes
from .abstract import (
    AbstractBaseMapper,
    AbstractBatchedBaseMapper,
    AbstractSingleBaseMapper,
)
from .interfaces import MapMethodInterfaceMixIn
from .types import TransformElementType

P = TypeVar("P", bound="PipelineFingerprintMixIn")


class PipelineFingerprintMixIn(AbstractBaseMapper):

    pipeline: Union["PipelineFingerprintMixIn", None]

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
        self.pipeline = None

    def chain(
        self: P, next_mapper: "PipelineFingerprintMixIn", inplace: bool = True
    ) -> P:
        """Create a pipeline by combining this mapper with another.

        Args:
            next_mapper (Mapper): The mapper to attach at the end of this
                mapper or pipeline.
            inplace (bool, optional): If True, the pipeline will be created
                in place. If False, a new pipeline will be created. Defaults
                to True.
        """

        # create a copy of this mapper before attaching it to the next mapper
        # unless inplace is True
        to_return = self if inplace else self.detach()

        # if the current mapper is already attached to a pipeline, we need
        # to recursively merge the pipelines; otherwise, we can just attach
        # the next mapper to it.
        if self.pipeline is not None:
            to_merge = self.pipeline.chain(next_mapper)
        else:
            to_merge = next_mapper

        # here's the part where we actually attach the next mapper!
        to_return.pipeline = to_merge

        return to_return

    def __lshift__(self, prev_mapper: P) -> P:
        """Create a pipeline by combining this mapper with another. This is
        equivalent to other.chain(self, inplace=False), but with notation
        self << other."""
        return prev_mapper.chain(self, inplace=False)

    def __rshift__(self: P, next_mapper: "PipelineFingerprintMixIn") -> P:
        """Create a pipeline by combining this mapper with another. This is
        equivalent to self.chain(other, inplace=False), but with notation
        self >> other."""
        return self.chain(next_mapper, inplace=False)

    def __repr__(self) -> str:
        """Return a string representation of this mapper."""
        r = f"{self.__class__.__name__}({self.fingerprint})"
        if self.pipeline is not None:
            r += f" >> {self.pipeline}"
        return r

    def __deepcopy__(
        self, memo: Optional[dict] = None
    ) -> "PipelineFingerprintMixIn":
        result = self.detach(memo)
        if self.pipeline is not None:
            result.pipeline = copy.deepcopy(self.pipeline, memo)

        return result

    def __getstate__(self) -> dict:
        """Return the state of this mapper for pickling."""
        state = {
            "__dict__": {
                k: v for k, v in self.__dict__.items() if k != "pipeline"
            },
            "__slots__": {
                k: getattr(self, k) for k in getattr(self, "__slots__", [])
            },
            # pipeline gets its own special treatment, which
            # is really just recursive pickling
            "pipeline": pickle.dumps(self.pipeline),
        }
        # if self.pipeline is not None:
        return state

    def __setstate__(self, state: dict) -> None:
        """Set the state of this mapper after unpickling."""
        self.__dict__.update(state.get("__dict__", {}))
        for k, v in state.get("__slots__", {}).items():
            setattr(self, k, v)
        self.pipeline = pickle.loads(state["pipeline"])

    def detach(self: P, memo: Optional[dict] = None) -> P:
        memo = memo or {}

        # create a new empty class
        cls = self.__class__
        result = cls.__new__(cls)

        # this dict helps with memoization in case of circular references
        memo[id(self)] = result

        for key in self.__dict__:
            if key == "pipeline":
                # don't copy the pipeline
                setattr(result, key, None)
            else:
                # copy the rest of the attributes
                setattr(result, key, copy.deepcopy(self.__dict__[key], memo))

        for slot in chain.from_iterable(
            getattr(s, "__slots__", []) for s in self.__class__.__mro__
        ):
            # copy the slots
            setattr(result, slot, copy.deepcopy(getattr(self, slot), memo))

        return result

    def __hash__(self) -> int:
        h = hashlib.md5()
        h.update(self.fingerprint.encode())
        if self.pipeline is not None:
            h.update(bytes_from_int(hash(self.pipeline)))
        return int_from_bytes(h.hexdigest().encode())

    def __eq__(self, other: object) -> bool:
        """Check if this mapper is equal to another."""
        if not isinstance(other, type(self)):
            return False

        return hash(self) == hash(other)

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

        # we need to inspect all frames in the frames where the various
        # __init__ methods in the MRO are called. This way we can get **all**
        # the arguments passed to the constructor, including those passed
        # through to super().__init__.
        stack_frames = [
            ExtInfo(arg_info=inspect.getargvalues(fr.frame), frame_info=fr)
            for fr in inspect.stack()
        ]

        # filter out the frames that are not __init__ methods here
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
            """Small helper function to get the name of the class from
            the frame info."""
            cls_ = frame_ext_info.arg_info.locals.get(
                "__class__", PipelineFingerprintMixIn
            )
            return f"{cls_.__module__}.{cls_.__name__}"

        # putting together the full init signature here
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
):
    """An abstract implementation of a Mapper that operates on a single
    element. All mappers that operate on a single element should subclass
    this class.

    Actual mapper implementations should override the transform method.
    The transform method should accept a single sample dictionary as input
    and return a single sample dictionary as output.
    """

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


class BatchedBaseMapper(
    MapMethodInterfaceMixIn,
    PipelineFingerprintMixIn,
    AbstractBatchedBaseMapper,
):
    """An abstract implementation of a Mapper that operates on a batch of
    elements. All mappers that operate on a batch should subclass this
    class.

    Actual mapper implementations should override the transform method.
    The transform method should accept a iterator of dictionaries as input,
    and return a iterator of dictionaries as output. The number of samples
    returned may be different from the number of samples in the input.
    """

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
