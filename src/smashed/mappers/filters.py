import operator
from typing import Any, Callable, Iterable, NamedTuple

from ..base import BatchedBaseMapper, TransformElementType
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

    def __init__(self, field_name: str, operator: str, value: Any):
        f"""
        Args:
            field_name (str): The name of the field to filter on.
            operator (str): The operator to use for filtering. Valid operators
                are {", ".join(VALID_OPERATIONS.keys())}
            value (Any): The value to compare against.
        """
        if operator not in VALID_OPERATIONS:
            raise ValueError(
                f"Invalid operator {operator}. Valid operators are "
                f"{', '.join(VALID_OPERATIONS.keys())}"
            )

        self.field_name = field_name
        self.operator = VALID_OPERATIONS[operator]
        self.value = value

        super().__init__(
            input_fields=[field_name],
            output_fields=[field_name],
        )

    def _single_op(self, value: Any, **_: Any) -> Any:
        return self.operator(value, self.value)

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

            if self._recursive_op(batch[self.field_name]):
                yield batch
