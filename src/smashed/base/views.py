from collections import abc
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    ItemsView,
    Iterable,
    KeysView,
    Tuple,
    Type,
    TypeVar,
    Union,
    ValuesView,
    cast,
)

K = TypeVar("K")
V = TypeVar("V")


class DEFAULT:
    ...


class DataRowView(abc.Mapping, Generic[K, V]):
    """A view of a row in a DataBatchView; supports dict-like access to
    the batch and in-place modification."""

    __slot__ = ("_dbv", "_idx")

    def __init__(self, dbv: "DataBatchView", idx: int):
        self._dbv = dbv
        self._idx = idx

    def __getitem__(self, key: K) -> V:
        return self._dbv._data[key][self._idx]

    def __setitem__(self, key: K, value: V):
        self._dbv._data[key][self._idx] = value

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DataRowView):
            return False

        for k in (*self.keys(), *other.keys()):
            if k not in other or k not in self:
                return False
            if self[k] != other[k]:
                return False

        return True

    def keys(self) -> KeysView[K]:
        return cast(KeysView[K], self._dbv._keys)

    def values(self) -> ValuesView[V]:
        return cast(
            ValuesView[V],
            (self._dbv._data[key][self._idx] for key in self._dbv._keys),
        )

    def items(self) -> ItemsView[K, V]:
        return cast(ItemsView[K, V], zip(self.keys(), self.values()))

    def update(self, other: Union["DataRowView[K, V]", Dict[K, V]]):
        for k, v in other.items():
            self[k] = v

    def __iter__(self) -> Generator[K, None, None]:
        return (k for k in self.keys())

    def __len__(self) -> int:
        return len(self._dbv._keys)

    @property
    def idx(self) -> int:
        return self._idx

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}"
            f"({self._idx}, {repr(self._dbv._data[self._idx])})"
        )

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}"
            f"({self._idx}, {str(self._dbv._data[self._idx])})"
        )

    def pop(self, key: K, default: Union[V, Type[DEFAULT]] = DEFAULT) -> V:
        if key in self:
            return self[key]
        elif default is not DEFAULT:
            return cast(V, default)
        else:
            raise KeyError(key)


D = TypeVar("D", bound=abc.MutableMapping)
T = TypeVar("T", bound="DataBatchView")


class DataBatchView(Generic[D, K, V]):
    """A view of a batch of data; supports list-like access to the batch
    and in-place modification via DataRowView."""

    __slots__ = ("_data", "_keys", "_len")

    _data: D
    _keys: Tuple[K, ...]
    _len_: int

    def __init__(self, data: D):
        self._data = data
        self._keys = tuple(data.keys())
        self._len = len(data[self._keys[0]])

    def keys(self) -> Iterable[K]:
        return self._data.keys()

    def values(self) -> Iterable[V]:
        return self._data.values()

    def items(self) -> Iterable[Tuple[K, V]]:
        return zip(self.keys(), self.values())

    def pop(self, key: K, default: Union[V, Type[DEFAULT]] = DEFAULT) -> V:
        if key in self._data:
            return self._data.pop(key)
        elif default is not DEFAULT:
            return cast(V, default)
        else:
            raise KeyError(key)

    def __getitem__(self, idx: int) -> DataRowView[K, V]:
        return DataRowView(self, idx)

    def __setitem__(
        self, idx: int, value: Union[DataRowView[K, V], Dict[K, V]]
    ):
        for (k, v) in value.items():
            self._data[k][idx] = v

    def __len__(self) -> int:
        return self._len

    def map(self: T, fn: Callable) -> T:
        return type(self)(fn(self._data))

    def update(
        self,
        other: Union[
            "DataBatchView[Any, K, V]",
            Iterable[Union[Dict[K, V], DataRowView[K, V]]],
        ],
    ):
        for i, row in enumerate(other):
            self[i] = row

    def __iter__(self) -> Generator[DataRowView[K, V], None, None]:
        for i in range(self._len):
            yield self[i]

    def orig(self) -> Any:
        return self._data

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data})"

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._data})"
