from collections.abc import Sequence as SequenceABC
from typing import Any, List, Sequence, Tuple, TypeVar, Union, cast

from typing_extensions import TypeAlias

T = TypeVar("T")

LocTupleType: TypeAlias = Tuple[int, int]
KeysType: TypeAlias = Union[LocTupleType, List["KeysType"]]
NestedSequenceType: TypeAlias = Union[
    Sequence[T], Sequence["NestedSequenceType[T]"]
]
NestedListType: TypeAlias = Union[List[T], List["NestedListType[T]"]]


def is_sequence_but_not_str(obj: Any) -> bool:
    """Check if an object is a sequence but not a string."""
    return isinstance(obj, SequenceABC) and not isinstance(obj, (str, bytes))


def flatten_with_indices(
    sequence: NestedSequenceType[T], __offset: int = 0
) -> Tuple[List[T], Union[KeysType, None]]:
    """Recursively flatten an iterable of iterables, returning both the
    flatten list, as well as the indices of the original list.

    Args:
        sequence (NestedSequenceType[T]): Either a sequence or a sequence
            of sequences; if a sequence of sequences, will be flattened.
        __offset (int, optional): Internal offset to keep track of the
            position in the flattened list. Defaults to 0; DO NOT CHANGE.

    Raises:
        ValueError: If the sequence contains both sequences and
            non-sequences.

    Returns:
        List[T]: The flattened list; if the original list was not nested,
            will be the same as the original list.
        Union[KeysType, None]: The indices of the original list; if the
            original list was not nested, will be None.
    """

    it = iter(sequence)
    flattened: list = []
    keys: list = []
    is_nested_sequence = is_already_flat = False

    while True:
        try:
            item = next(it)
        except StopIteration:
            break

        if is_sequence_but_not_str(item):
            if is_already_flat:
                raise ValueError(
                    "Cannot mix sequences and non-sequences when flattening."
                )
            is_nested_sequence = True

            offset = len(flattened) + __offset
            # manual casting bc we know this is a sequence (see function
            # is_sequence_but_not_str) but if we don't cast mypy is going
            # to complain.
            item = cast(NestedSequenceType[T], item)

            # must use type: ignore here because mypy doesn't like using
            # the __offset kwarg (which is a good idea in general but
            # we nee to use it during recursive calls)
            sub_flattened, sub_keys = flatten_with_indices(  # type: ignore
                sequence=item, __offset=offset
            )

            if sub_keys is None:
                sub_keys = (offset, offset + len(sub_flattened))

            keys.append(sub_keys)
            flattened.extend(sub_flattened)
        else:
            if is_nested_sequence:
                raise ValueError(
                    "Cannot mix sequences and non-sequences when flattening."
                )
            is_already_flat = True

            flattened.append(item)

    return flattened, (keys or None)


def reconstruct_from_indices(
    flattened: List[T], keys: Union[KeysType, None]
) -> NestedListType[T]:
    """Recursively reconstruct a list from a flattened list and the keys that
    were returned from recursively_flatten_with_indices.

    Args:
        flattened (List[T]): A flat list of items.

    """

    if keys is None:
        return flattened

    reconstructed: list = []
    for key in keys:
        if isinstance(key, list):
            reconstructed.append(reconstruct_from_indices(flattened, key))
        elif isinstance(key, tuple):
            start, end = key
            reconstructed.append(flattened[start:end])
        else:
            raise ValueError(
                f"Invalid key type: expected tuple or list, got {type(key)}"
            )

    return reconstructed
