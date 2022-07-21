
from typing import Any, Optional, Sequence, Callable

from ..base import BaseMapper, TransformElementType, BaseDataset


class ChangeFieldsMapper(BaseMapper):
    def __init__(self,
                 keep_fields:  Optional[Sequence[str]] = None,
                 drop_fields: Optional[Sequence[str]] = None):

        # xor between keep_fields and remove_fields
        if (keep_fields is not None and drop_fields is not None) or \
                (keep_fields is None and drop_fields is None):
            raise ValueError('Must specify `keep_fields` or `drop_fields`')

        self.input_fields = drop_fields or []
        self.output_fields = keep_fields or []

        self.batched = False

    def map(self,
            dataset: BaseDataset,
            **map_kwargs: Any) -> BaseDataset:
        map_kwargs = {'remove_columns': list(dataset.features.keys()),
                      **map_kwargs}
        return super().map(dataset, **map_kwargs)

    def transform(self, data: TransformElementType) -> TransformElementType:
        if self.input_fields:
            new_data = {k: v for k, v in data.items()
                        if k not in self.input_fields}

        elif self.output_fields:
            new_data = {k: data[k] for k in self.output_fields}

        else:
            raise ValueError('Must specify `keep_fields` or `drop_fields`')

        return new_data


class MakeFieldMapper(BaseMapper):
    def __init__(
        self,
        field_name: str,
        value: Optional[Any] = None,
        shape_like: Optional[str] = None,
        value_fn: Optional[Callable[[TransformElementType], Any]] = None
    ):
        self.input_fields = []
        self.output_fields = [field_name]
        self.batched = False

        if value_fn is None:
            if value is None:
                raise ValueError('Must specify `value` or `value_fn`')

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
