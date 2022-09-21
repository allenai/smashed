from typing import TYPE_CHECKING, Any, Dict, TypeVar, Union

from necessary import necessary
from trouting import trouting

from ..base import SingleBaseMapper, TransformElementType

with necessary("datasets", soft=True) as HUGGINGFACE_DATASET_AVAILABLE:
    if HUGGINGFACE_DATASET_AVAILABLE or TYPE_CHECKING:
        from datasets.arrow_dataset import Dataset
        from datasets.iterable_dataset import IterableDataset

        HuggingFaceDataset = TypeVar(
            "HuggingFaceDataset", Dataset, IterableDataset
        )
        from datasets.features import features


HF_CAST_DICT = {
    int: "int64",
    float: "float32",
    bool: "bool",
    str: "string",
}


class RecurseOpMixIn:
    def _single_op(self, value: Any, **_: Any) -> Any:
        raise NotImplementedError

    def _recursive_op(self, value: Any, **kwargs: Any) -> Any:
        if isinstance(value, list):
            return [self._recursive_op(value=v, **kwargs) for v in value]
        elif isinstance(value, dict):
            return {
                k: self._recursive_op(value=v, **kwargs)
                for k, v in value.items()
            }
        else:
            return self._single_op(value=value, **kwargs)


class CastMapper(SingleBaseMapper, RecurseOpMixIn):
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

    def _single_op(self, value: Any, type_: type) -> Any:  # type: ignore
        try:
            return type_(value)
        except ValueError:
            raise ValueError(f"Could not cast value {value} to type {type_}")

    def transform(self, data: TransformElementType) -> TransformElementType:
        return {
            k: (
                self._recursive_op(value=v, type_=self.cast_map[k])
                if k in self.cast_map
                else v
            )
            for k, v in data.items()
        }

    if HUGGINGFACE_DATASET_AVAILABLE:

        def _build_feature_definition(
            self,
            def_n: Union[
                dict, features.ClassLabel, features.Sequence, features.Value
            ],
            type_: type,
        ) -> Union[features.ClassLabel, features.Sequence, features.Value]:
            """A helper function to build a new feature definition in case
            the dataset is a HuggingFace dataset.

            Args:
                def_n (Union[ features.ClassLabel, features.Sequence, features.
                    Value ]): The current feature definition from
                    datasets.features dictionary.
                type_ (type): The type to cast the feature to.

            Returns:
                Union[ features.ClassLabel, features.Sequence,
                    features.Value ]: The new feature definition.
            """

            # TODO[soldni]: document better!

            if (
                t_str := HF_CAST_DICT.get(type_, None)  # pyright: ignore
            ) is None:
                raise ValueError(
                    f"Unsupported type {type_} for HuggingFace Dataset"
                )

            if isinstance(def_n, features.Sequence):
                if isinstance(def_n.feature, dict):
                    new_feature = {
                        k: self._build_feature_definition(v, type_)
                        for k, v in def_n.feature.items()
                    }
                else:
                    new_feature = self._build_feature_definition(
                        def_n.feature, type_
                    )

                new_definition = features.Sequence(feature=new_feature)

            elif isinstance(def_n, features.ClassLabel):
                new_names = list(set(type_([n for n in def_n.names])))
                new_definition = features.ClassLabel(
                    names=new_names,
                    num_classes=len(new_names),
                )

            elif isinstance(def_n, features.Value):
                new_definition = features.Value(dtype=t_str)
            else:
                raise ValueError(f"Unsupported feature definition {def_n}")

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
                        def_n=dataset.features[field_name], type_=field_type
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

    def _single_op(self, value: Any, **_: Any) -> Any:  # type: ignore
        return int(value > self.threshold)


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

    def _single_op(self, value: Any, **_: Any) -> Any:  # type: ignore
        return self.lookup_table[value]


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

    def _single_op(self, value: Any, **_: Any) -> Any:  # type: ignore
        return [1 if i == value else 0 for i in range(self.num_classes)]
