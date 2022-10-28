import re
from ast import literal_eval
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

KeyFr = Union[List["KeyFr"], Tuple["KeyFr", ...], str, int]  # type: ignore
KeyType = Union[List[KeyFr], Tuple[KeyFr, ...]]  # type: ignore
DataType = Union[dict, list]
D = TypeVar("D", bound=DataType)


class MISSING:
    def __init__(self, _: KeyFr) -> None:
        raise RuntimeError("Do not instantiate MISSING; it is a singleton.")


class Nested:
    """
    A tool to operate on nested data structures.

    Key grammar is as follows:
    - A key is a sequence of key fragments.
    - Fragments are separated by full stops, e.g. `a.b.c`.
    - A key fragment is either a string, an integer, or a list containing
        a sequence of key fragments.
    - A string key fragment indicates a dictionary key; in case the key
        contains reserved characters, it must be quoted, e.g. `a."b.c"`.
    - An integer key fragment indicates a list index. Negative indices are
        supported, e.g. `a.-1`.
    - A list key indicates that all elements of a list must be processed.
        list key can use either square brackets or parentheses, e.g. so the
        keys `a.[b.c]` and `a.(b.c)` are equivalent. A list key can be
        nested, e.g. `a.[b.[c.d]]`.

    Example:

    Given the following data structure:

    data = {
        "a": {
            "b": 3,
            "c": [{"d": 4}, {"d": 5}],
            "e.f": {"g": [6, 7]},
        }
    }

    The following keys are valid:
        - `a.b` (selects 3)
        - `a.c` (selects [{"d": 4}, {"d": 5}])
        - `a.c.[d]` (selects [4, 5])
        - `a.c.-1.d` (selects 5)
        - `a."e.f".g` (selects [6, 7])
        - `a."e.f".g.[]` (selects [6, 7])
    """

    def __init__(self, key: KeyType):
        """Create a Nested object from an already parsed key."""

        if isinstance(key, str):
            name = {type(self).__name__}
            raise TypeError(
                f"{name}.__init__ expects a parsed key, not a string. Use "
                f"{name}.from_str() to create a {name} object from a string."
            )

        self.key = key

    def __len__(self) -> int:
        """Number of top-level steps in the key."""
        return len(self.key)

    def __getitem__(self, idx: int) -> KeyFr:
        return self.key[idx]

    @classmethod
    def to_str(cls, key: KeyType) -> str:
        """Returns the key of this Nested object as a string."""

        return ".".join(
            f"[{cls.to_str(k)}]"
            if isinstance(k, (list, tuple))
            else (
                # escape string if it contains any reserved characters
                f'"{k}"'
                if (
                    # k might be a number so we need to check for that
                    isinstance(k, str)
                    and re.search(r"[\.\]\[\)\(\'\"]", k)
                )
                else str(k)
            )
            for k in key
        )

    def __str__(self) -> str:
        return self.to_str(self.key)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.key})"

    @classmethod
    def from_str(cls, key: str) -> "Nested":
        return cls(key=cls.parse_key(key))

    @classmethod
    def _nested_replace(
        cls, parsed: KeyType, quoted: Dict[str, str]
    ) -> KeyType:
        """Recursive function that looks for placeholders and replaces them
        with text in quoted.

        Args:
            parsed (Sequence[Any]): The parsed key where we look
                for placeholders.
            quoted (Dict[str, glom.Path]): A dictionary where we look for
            placeholders.
        """
        parsed = type(parsed)(
            # its ok if we look for ints in a dict of str keys, bc they will
            # naturally fail and the default will be used instead; the cast
            # is trick mypy and make it stop complaining.
            quoted.get(cast(str, p), p)
            if isinstance(p, (str, int))
            else cls._nested_replace(parsed=p, quoted=quoted)
            for p in parsed
        )
        return parsed

    @classmethod
    def parse_key(cls, key: str) -> KeyType:
        # save original key in case we need to raise an error
        og_key = key

        # anything that is in placeholders will be saved for later
        placeholders: Dict[str, str] = {}

        # start by replacing quoted parts with a placeholder;
        for match in re.finditer(r"[\"\'].*[\"\']", key):
            # Wrapping in glom.Path is required to ensure that each is treated
            # as a single key name by glom even if they contain dots or other
            # special characters, e.g. "a.b" for {"a.b": 1}.
            idx = f"__{len(placeholders)}__"
            text = match.group(0)
            placeholders[idx] = text[1:-1]
            key = key.replace(text, idx)

        # add left quote to all keys
        key = re.sub(r"(^|\.|\(|\[)([^\.\]\[\)\(]+?)", r'\1"\2', key)

        # add right quote to all keys
        key = re.sub(r"([^\.\]\[\)\(]+?)($|\.|\]|\))", r'\1"\2', key)

        # if any number got quoted, we need to remove the quotes
        key = re.sub(r"\"(\d+)\"", r"\1", key)

        # replace all dots with commas so that we can eval the result
        key = key.replace(".", ",")

        # eval the result using literal_eval
        try:
            parsed = literal_eval(f"{key}")
        except Exception as e:
            raise SyntaxError(f"Could not parse key {og_key}") from e

        if isinstance(parsed, str):
            parsed = (parsed,)

        # Put any quoted segments back
        return cls._nested_replace(parsed=parsed, quoted=placeholders)

    @classmethod
    def _edit(
        cls,
        key: Union[List[Any], Tuple[Any, ...]],
        data: DataType,
        fn: Callable,
        missing: Callable[[KeyFr], Any] = MISSING,
    ):
        this, *rest = key

        if isinstance(this, (tuple, list)):
            if len(rest) > 0:
                raise ValueError("No keys should follow a nested key")

            if not isinstance(data, list):
                if missing is MISSING:
                    raise ValueError(
                        f"Expected list for key {this}, got {type(data)}"
                    )
                else:
                    return missing(rest)

            elif len(this) > 0:
                for elem in data:
                    cls._edit(key=this, data=elem, fn=fn, missing=missing)
            else:
                for i, elem in enumerate(data):
                    data[i] = fn(elem)

        elif isinstance(this, int):
            if not isinstance(data, list):
                if missing is MISSING:
                    raise ValueError(
                        f"Expected list for key {this}, got {type(data)}"
                    )

            elif this >= len(data) or this < -len(data):
                if missing is MISSING:
                    raise IndexError(
                        f"Index {this} out of range for list {data}  "
                        f"of length {len(data)}"
                    )
                else:
                    return missing(rest)

            if len(rest) > 0:
                cls._edit(key=rest, data=data[this], fn=fn, missing=missing)
            else:
                data[this] = fn(data[this])

        elif isinstance(this, str):
            if not isinstance(data, dict):
                if missing is MISSING:
                    raise ValueError(
                        f"Expected dict for key {this}, got {type(data)}"
                    )
                else:
                    return missing(rest)

            elif this not in data:
                if missing is MISSING:
                    raise KeyError(f"Key {this} not found in dict {data}")
                else:
                    return missing(rest)

            if len(rest) > 0:
                cls._edit(key=rest, data=data[this], fn=fn, missing=missing)
            else:
                data[this] = fn(data[this])
        else:
            raise ValueError(f"Invalid key {key}")

    @staticmethod
    def none(_: KeyFr) -> None:
        return None

    def edit(
        self,
        data: D,
        fn: Callable,
        inplace: bool = True,
        missing: Optional[Callable[[KeyFr], Any]] = MISSING,
    ) -> D:
        missing = missing or self.none
        data = deepcopy(data) if not inplace else data
        self._edit(key=self.key, data=data, fn=fn, missing=missing)
        return data

    @classmethod
    def _get(
        cls,
        key: Union[List[Any], Tuple[Any, ...]],
        data: DataType,
        flat: bool = False,
        missing: Callable[[KeyFr], Any] = MISSING,
    ) -> Any:
        if len(key) == 0:
            return data

        this, *rest = key

        if isinstance(this, (tuple, list)):
            if len(rest) > 0:
                raise ValueError("No keys should follow a nested key")

            if not isinstance(data, list):
                if missing is MISSING:
                    raise ValueError(
                        f"Expected list for key {this}, got {type(data)}"
                    )
                else:
                    data = [missing(rest)]

            out = [
                cls._get(key=this, data=elem, flat=flat, missing=missing)
                for elem in data
            ]
            if flat:
                out = [
                    elem
                    for sub in out
                    for elem in (sub if isinstance(sub, list) else (sub,))
                ]
            return out

        elif isinstance(this, int):
            if not isinstance(data, list):
                if missing is MISSING:
                    raise ValueError(
                        f"Expected list for key {this}, got {type(data)}"
                    )
                else:
                    data = [missing(rest)] * (this + 1 if this >= 0 else -this)

            if this >= len(data) or this < -len(data):
                if missing is MISSING:
                    raise IndexError(
                        f"Index {this} out of range for list {data}  "
                        f"of length {len(data)}"
                    )
                else:
                    data = data + (
                        [missing(rest)] * (this + 1 if this >= 0 else -this)
                    )

            out = cls._get(
                key=rest, data=data[this], flat=flat, missing=missing
            )
            return out if flat else [out]

        elif isinstance(this, str):
            if not isinstance(data, dict):
                if missing is MISSING:
                    raise ValueError(
                        f"Expected dict for key {this}, got {type(data)}"
                    )
                else:
                    data = {this: missing(rest)}

            if this not in data:
                if missing is MISSING:
                    raise KeyError(f"Key {this} not found in dict {data}")
                else:
                    data[this] = missing(rest)

            out = cls._get(
                key=rest, data=data[this], flat=flat, missing=missing
            )
            return out if flat else {this: out}
        else:
            raise ValueError(f"Invalid key {key}")

    def copy(
        self, data: D, missing: Optional[Callable[[KeyFr], Any]] = MISSING
    ) -> D:
        missing = missing or self.none
        return self._get(key=self.key, data=data, flat=False, missing=missing)

    def select(
        self, data: D, missing: Optional[Callable[[KeyFr], Any]] = MISSING
    ) -> D:
        missing = missing or self.none
        return self._get(key=self.key, data=data, flat=True, missing=missing)
