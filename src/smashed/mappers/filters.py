from collections import abc
import operator
from typing import Any, Callable, Dict, Iterable, NamedTuple, Tuple

from ..base.mappers import BatchedBaseMapper
from ..base.types import TransformElementType
from .types import RecurseOpMixIn

VALID_OPERATIONS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "%": operator.mod,
    "^": operator.xor,
    "**": operator.pow,
    "<<": operator.lshift,
    ">>": operator.rshift,
    "&": operator.and_,
    "|": operator.or_,
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "is": operator.is_,
    "is not": operator.is_not,
    "in": operator.contains,
    "not in": lambda x, y: not operator.contains(x, y),
}


class FuncVal(NamedTuple):
    func_: Callable[[Any, Any], Any]
    value: Any


class FilterMapper(BatchedBaseMapper, RecurseOpMixIn):
    """A mapper that filters elements from a batch."""

    def __init__(self, fields_filters: Dict[str, Tuple[str, Any]]):
        f"""
        Args:
            fields_filters (Dict[str, Tuple[str, Any]]): A dictionary of fields
                and their filters. Filters are a (operator, value) tuple.
                The operator is a string from {','.join(VALID_OPERATIONS)}
                and the value is the value to compare against.
        """

        assert isinstance(fields_filters, dict) and all(
            # we are actually ok with sequence as long as has a length of 2,
            # but asking for tuple makes type annotation more concise
            isinstance(v, abc.Sequence) and len(v) == 2
            for v in fields_filters.values()
        ), "fields_filters must be a Dict[str, Tuple[str, Any]]"

        try:
            self.fields_filters = {
                k: FuncVal(func_=VALID_OPERATIONS[v[0]], value=v[1])
                for k, v in fields_filters.items()
            }
        except KeyError:
            raise ValueError(
                "Invalid operator in filter. Valid operators are "
                f"{','.join(VALID_OPERATIONS)}"
            )

        super().__init__(
            input_fields=list(fields_filters.keys()),
            output_fields=list(fields_filters.keys()),
        )

    def _single_op(  # type: ignore
        self, value: Any, func_val: FuncVal
    ) -> bool:
        return bool(func_val.func_(value, func_val.value))

    def _recursive_op(self, value: Any, **kwargs: Any) -> Any:
        recursed = super()._recursive_op(value, **kwargs)
        if isinstance(recursed, list):
            return all(recursed)
        if isinstance(recursed, dict):
            return all(recursed.values())
        else:
            return recursed

    def transform(
        self, data: Iterable[TransformElementType]
    ) -> Iterable[TransformElementType]:
        for batch in data:
            filter_pass = all(
                self._recursive_op(value=batch[field], func_val=func_val)
                for field, func_val in self.fields_filters.items()
            )
            if filter_pass:
                yield batch
