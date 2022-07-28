from typing import Any, Callable, List, Optional, TypeVar

from ..base.dataset import BaseDataset
from ..base.mapper import SingleBaseMapper
from ..base.types import TransformElementType

D = TypeVar("D", bound="BaseDataset")


class ChangeFieldsMapper(SingleBaseMapper):
    def __init__(
        self,
        keep_fields: Optional[List[str]] = None,
        drop_fields: Optional[List[str]] = None,
    ):
        """Mapper that removes some of the fields in a dataset.
        Either `keep_fields` or `drop_fields` must be specified, but not both.

        Args:
            keep_fields (List[str]): Fields to keep, all other fields
                are dropped. Defaults to [].
            drop_fields (List[str]): Fields to drop, all other fields
                are kept. Defaults to [].
        """

        # xor between keep_fields and remove_fields
        if (keep_fields is not None and drop_fields is not None) or (
            keep_fields is None and drop_fields is None
        ):
            raise ValueError("Must specify `keep_fields` or `drop_fields`")

        super().__init__(input_fields=drop_fields, output_fields=keep_fields)

    def map(self, dataset: D, **map_kwargs: Any) -> D:
        map_kwargs = {
            "remove_columns": list(dataset.features.keys()),
            **map_kwargs,
        }
        return super().map(dataset, **map_kwargs)

    def transform(self, data: TransformElementType) -> TransformElementType:
        if self.input_fields:
            new_data = {
                k: v for k, v in data.items() if k not in self.input_fields
            }

        elif self.output_fields:
            new_data = {k: data[k] for k in self.output_fields}

        else:
            raise ValueError("Must specify `keep_fields` or `drop_fields`")

        return new_data


class MakeFieldMapper(SingleBaseMapper):
    def __init__(
        self,
        field_name: str,
        value: Optional[Any] = None,
        shape_like: Optional[str] = None,
        value_fn: Optional[Callable[[TransformElementType], Any]] = None,
    ):
        """Mapper that adds a new field to a dataset.
        Either `value` or `value_fn` must be specified, but not both.
        `value` is a constant value to assign to the new field, while
        `value_fn` is a function that takes the full sample as input and
        returns a value to assign to the new field.

        Args:
            field_name (str): Name of the new field.
            value (Optional[Any], optional): Value to assign to the new field.
            shape_like (Optional[str], optional): If a fixed value is provided,
                this existing field that will be used to determine the shape of
                the new field. Defaults to None.
            value_fn (Optional[Callable[[TransformElementType], Any]], optional):
                Function to call to assign a value to the new field.
                Defaults to None.
        """
        super().__init__(output_fields=[field_name])

        if value_fn is None:
            if value is None:
                raise ValueError("Must specify `value` or `value_fn`")

            def _value_fn(data: TransformElementType) -> Any:
                if shape_like is not None:
                    return [value for _ in data[shape_like]]
                else:
                    return value

            # doing the two assignment separately to avoid linter
            # error about redefinition of `value_fn` via def
            value_fn = _value_fn

        self.value_fn = value_fn

    def transform(self, data: TransformElementType) -> TransformElementType:
        data[self.output_fields[0]] = self.value_fn(data)
        return data
