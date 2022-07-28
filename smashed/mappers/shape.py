from itertools import chain
from typing import Dict

from ..base.mapper import SingleBaseMapper
from ..base.types import Features, FeatureType, TransformElementType


class FlattenMapper(SingleBaseMapper):
    def __init__(self, field: str) -> None:
        super().__init__(input_fields=[field], output_fields=[field])

    def transform(self, data: TransformElementType) -> TransformElementType:
        field_name, *_ = self.input_fields

        flattened_field = data[field_name]

        if len(flattened_field) > 0:
            while isinstance(flattened_field[0], list):
                flattened_field = list(chain.from_iterable(flattened_field))

        return {field_name: flattened_field}


class BinarizerMapper(SingleBaseMapper):
    __value_type__: type = str
    __sequence_type__: type = list

    def __init__(self, field: str, threshold: float) -> None:
        super().__init__(input_fields=[field], output_fields=[field])
        self.threshold = threshold

    def cast_columns(self, features: Features) -> Dict[str, FeatureType]:
        field_name, *_ = self.input_fields
        if isinstance(features[field_name], self.__sequence_type__):
            new_field = self.__sequence_type__(self.__value_type__("int64"))
        else:
            new_field = self.__value_type__("int64")

        return {field_name: new_field}

    def transform(self, data: TransformElementType) -> TransformElementType:
        field_name, *_ = self.input_fields

        if isinstance(data[field_name], list):
            return {
                field_name: [
                    1 if v > self.threshold else 0 for v in data[field_name]
                ]
            }
        else:
            return {field_name: 1 if data[field_name] > self.threshold else 0}
