from typing import TYPE_CHECKING, Any, Dict, TypeVar, Union

from necessary import necessary
from trouting import trouting

from ..base.mappers import SingleBaseMapper
from ..base.types import TransformElementType

with necessary("datasets", soft=True) as HUGGINGFACE_DATASET_AVAILABLE:
    if HUGGINGFACE_DATASET_AVAILABLE or TYPE_CHECKING:
        from datasets.arrow_dataset import Dataset
        from datasets.iterable_dataset import IterableDataset

        HuggingFaceDataset = TypeVar(
            "HuggingFaceDataset", Dataset, IterableDataset
        )
        from datasets.features import features


class CastMapper(SingleBaseMapper):
    """Casts one or more fields in a dataset to a new type."""

    def __init__(self, cast_map: Dict[str, type]):
        """Initializes a CastMapper.

        Args:
            cast_map (Dict[str, type]): A mapping from field names to the
                type to cast them to. In case a dataset is a HuggingFace
                dataset, the type will be converted to the appropriate value
                for the datasets.Value class.
        """
        self.cast_map = cast_map
        super().__init__(
            input_fields=list(cast_map.keys()),
            output_fields=list(cast_map.keys()),
        )

    @trouting
    def map(  # type: ignore
        self,
        dataset: Any,
        **map_kwargs: Any,
    ) -> Any:
        # we need this map to be able to add the new interface below
        # and handle types for which we don't have a new interface but our
        # parent class has one
        return super().map(dataset, **map_kwargs)

    def transform(self, data: TransformElementType) -> TransformElementType:
        return {
            k: self.cast_map[k](v) if k in self.cast_map else v
            for k, v in data.items()
        }

    if HUGGINGFACE_DATASET_AVAILABLE:

        def _build_feature_definition(
            self,
            defn: Union[
                features.ClassLabel, features.Sequence, features.Value
            ],
            typ_: type,
        ) -> Union[features.ClassLabel, features.Sequence, features.Value]:

            if isinstance(typ_, int):
                t_str = "int64"
            elif isinstance(typ_, float):
                t_str = "float32"
            elif isinstance(typ_, bool):
                t_str = "bool"
            elif isinstance(typ_, str):
                t_str = "string"
            else:
                raise ValueError(
                    f"Unsupported type {typ_} for HuggingFace Dataset"
                )

            if isinstance(defn, features.Sequence):
                new_definition = features.Sequence(
                    feature={
                        k: self._build_feature_definition(defn=v, typ_=typ_)
                        for k, v in defn.feature.items()
                    }
                )

            elif isinstance(defn, features.ClassLabel):
                new_names = list(set(typ_([n for n in defn.names])))
                new_definition = features.ClassLabel(
                    names=new_names,
                    num_classes=len(new_names),
                )

            elif isinstance(defn, features.Value):
                new_definition = features.Value(type=t_str)  # type: ignore
            else:
                raise ValueError(f"Unsupported feature definition {defn}")

            return new_definition

        @map.add_interface(dataset=(Dataset, IterableDataset))
        def map_huggingface_dataset(
            self,
            dataset: HuggingFaceDataset,
            **map_kwargs: Any,
        ) -> HuggingFaceDataset:
            dataset = super().map(dataset, **map_kwargs)
            for field_name, field_type in self.cast_map.items():
                dataset = dataset.cast_column(
                    field_name,
                    self._build_feature_definition(
                        defn=dataset.features[field_name], typ_=field_type
                    ),
                )
            return dataset


class BinarizerMapper(CastMapper):
    """Binarizes a field in a dataset."""

    def __init__(self, field: str, threshold: float) -> None:
        """Initializes a BinarizerMapper.

        Args:
            field (str): The field to binarize.
            threshold (float): The threshold to use for binarization.

        """
        super().__init__(cast_map={field: int})
        self.threshold = threshold

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


class LookupMapper(CastMapper):
    def __init__(self, field_name: str, lookup_table: Dict[Any, Any]):

        source_types = set([type(k) for k in lookup_table.keys()])
        target_types = set([type(v) for v in lookup_table.values()])

        if len(source_types) > 1:
            raise ValueError(
                "Lookup source values must be of the same type, "
                f"but got {source_types}"
            )
        if len(target_types) > 1:
            raise ValueError(
                "Lookup target values must be of the same type, "
                f"but got {target_types}"
            )

        super().__init__(cast_map={field_name: list(target_types)[0]})
        self.field_name = field_name
        self.lookup_table = lookup_table

    def transform(self, data: TransformElementType) -> TransformElementType:
        return {
            **data,
            self.field_name: self.lookup_table[data[self.field_name]],
        }


class OneHotMapper(CastMapper):
    """One-hot encodes a field in a dataset."""

    def __init__(self, num_classes: int, field_name: str) -> None:
        """Initializes a OneHotMapper.

        Args:
            num_classes (int): The number of classes.
            field (str): The field to one-hot encode.

        """
        super().__init__(cast_map={field_name: int})
        self.field_name = field_name
        self.num_classes = num_classes

    def transform(self, data: TransformElementType) -> TransformElementType:
        return {
            **data,
            self.field_name: [
                1 if i == data[self.field_name] else 0
                for i in range(self.num_classes)
            ],
        }
