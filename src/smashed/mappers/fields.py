from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypeVar

from necessary import necessary
from torch._utils import classproperty

from ..base import SingleBaseMapper, TransformElementType

with necessary("datasets", soft=True) as HUGGINGFACE_DATASET_AVAILABLE:
    if HUGGINGFACE_DATASET_AVAILABLE or TYPE_CHECKING:
        from datasets.arrow_dataset import Dataset
        from datasets.iterable_dataset import IterableDataset

        HuggingFaceDataset = TypeVar(
            "HuggingFaceDataset", Dataset, IterableDataset
        )


class ChangeFieldsMapper(SingleBaseMapper):
    """Mapper that removes some of the fields in a dataset.
    Either `keep_fields` or `drop_fields` must be specified, but not both."""

    @classproperty
    def always_remove_columns(cls) -> bool:
        return True

    def __init__(
        self,
        keep_fields: Optional[List[str]] = None,
        drop_fields: Optional[List[str]] = None,
    ):
        """
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


class RenameFieldsMapper(SingleBaseMapper):
    """Mapper that renames some of the fields batch"""

    @classproperty
    def always_remove_columns(cls) -> bool:
        return True

    def __init__(
        self, rename_fields_map: Dict[str, str], remove_rest: bool = False
    ):
        """
        Args:
            rename_fields_map (Dict[str, str]): Mapping from old field name
                to new field name.
            remove_rest (bool, optional): Whether to remove fields that are
                not in the rename_fields_map. Defaults to False.
        """

        self.rename_fields_map = rename_fields_map
        self.remove_rest = remove_rest
        super().__init__(
            input_fields=list(rename_fields_map.keys()),
            output_fields=list(rename_fields_map.values()),
        )

    def transform(self, data: TransformElementType) -> TransformElementType:
        return {
            self.rename_fields_map.get(k, k): v
            for k, v in data.items()
            if k in self.rename_fields_map or not self.remove_rest
        }


class MakeFieldMapper(SingleBaseMapper):
    def __init__(
        self: "MakeFieldMapper",
        field_name: str,
        value: Any,
        shape_like: Optional[str] = None,
    ):
        """Mapper that adds a new field to a dataset.

        Args:
            field_name (str): Name of the new field.
            value (Optional[Any]): Value to assign to the new field.
            shape_like (Optional[str], optional): If a fixed value is provided,
                this existing field that will be used to determine the shape of
                the new field. Defaults to None.

        """
        super().__init__(output_fields=[field_name])
        self.value = value
        self.shape_like = shape_like

    def transform(self, data: TransformElementType) -> TransformElementType:
        if self.shape_like is not None:
            new_value = [self.value for _ in data[self.shape_like]]
        else:
            new_value = self.value

        data[self.output_fields[0]] = new_value
        return data
