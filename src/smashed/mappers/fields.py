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
        raise_on_missing: bool = True,
    ):
        """
        Args:
            keep_fields (List[str]): Fields to keep, all other fields
                are dropped. Defaults to [].
            drop_fields (List[str]): Fields to drop, all other fields
                are kept. Defaults to [].
            raise_on_missing (bool): Whether to raise an error if a field
                is missing. Defaults to True.
        """

        # xor between keep_fields and remove_fields
        if (keep_fields is not None and drop_fields is not None) or (
            keep_fields is None and drop_fields is None
        ):
            raise ValueError("Must specify `keep_fields` or `drop_fields`")

        self.keep_fields = dict.fromkeys(keep_fields) if keep_fields else None
        self.drop_fields = dict.fromkeys(drop_fields) if drop_fields else None

        super().__init__(
            input_fields=drop_fields if raise_on_missing else None,
            output_fields=keep_fields if raise_on_missing else None,
        )

    def transform(self, data: TransformElementType) -> TransformElementType:
        if self.drop_fields:
            new_data = {
                k: v for k, v in data.items() if k not in self.drop_fields
            }

        elif self.keep_fields:
            new_data = {k: data[k] for k in data if k in self.keep_fields}

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
    """Mapper that adds a new field to a dataset."""

    def __init__(
        self: "MakeFieldMapper",
        field_name: str,
        value: Any,
        shape_like: Optional[str] = None,
    ):
        """Args:
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


class EnumerateFieldMapper(SingleBaseMapper):
    """Enumerate values in a field; optionally assigning the same id to
    repeated values."""

    def __init__(
        self,
        field_to_enumerate: str,
        destination_field: Optional[str] = None,
        same_id_for_repeated: bool = True,
    ):
        """Args:
        field_to_enumerate (str): Name of the field to enumerate.
        destination_field (str, optional): Name of the field where the
            enumeration of samples will be stored. If None, the enumeration
            will replace the original field. Defaults to None.
        same_id_for_repeated (bool, optional): Whether to assign the same
            id to repeated values. Requires value in the field to
            be hashable. Defaults to True.
        """
        self.enum_field = field_to_enumerate
        self.dest_field = destination_field or field_to_enumerate
        self.same_id_for_repeated = same_id_for_repeated
        self._init_memory()

        super().__init__(
            input_fields=[self.enum_field], output_fields=[self.dest_field]
        )

    def _init_memory(self):
        """Initializes counters to keep track of enumeration."""
        self.memory: Dict[Any, int] = {}
        self.count: int = 0

    def __getstate__(self) -> dict:
        out = super().__getstate__()

        # do not store enumerations when pickling the mapper
        out["__dict__"].pop("memory")
        out["__dict__"].pop("count")
        return out

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        # reinitialize counters when unpickling the mapper
        self._init_memory()

    def transform(self, element: TransformElementType) -> TransformElementType:
        if self.same_id_for_repeated:
            try:
                i = self.memory.setdefault(
                    element[self.enum_field], len(self.memory)
                )
            except TypeError as e:
                raise TypeError(
                    f"Could not enumerate field `{self.enum_field}` "
                    "because it is not hashable."
                ) from e
        else:
            i = self.count
            self.count += 1

        element[self.dest_field] = i
        return element
