import inspect
import json
from collections import abc
from csv import DictReader
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
    cast,
)

from necessary import necessary

from ..base import BatchedBaseMapper, TransformElementType

with necessary("datasets", soft=True) as HUGGINGFACE_DATASET_AVAILABLE:
    if HUGGINGFACE_DATASET_AVAILABLE or TYPE_CHECKING:
        from datasets.arrow_dataset import Dataset
        from datasets.combine import concatenate_datasets, interleave_datasets
        from datasets.iterable_dataset import IterableDataset
        from datasets.load import load_dataset

        HuggingFaceDataset = TypeVar(
            "HuggingFaceDataset", Dataset, IterableDataset
        )

with necessary("smart_open", soft=True) as SMART_OPEN_AVAILABLE:
    if SMART_OPEN_AVAILABLE:
        from smart_open import open


class HuggingFaceDatasetLoaderMapper(BatchedBaseMapper):
    def __init__(
        self,
        combine_strategy: Union[
            Literal["concatenate"], Literal["interleave"]
        ] = "concatenate",
        fields_to_keep: Optional[List[str]] = None,
        **kwargs,
    ):
        if not HUGGINGFACE_DATASET_AVAILABLE:
            raise ImportError(
                "HuggingFace datasets is not installed. To use this mapper, "
                "please install it with `pip install datasets`."
            )

        valid_strategies = {"concatenate", "interleave"}
        if combine_strategy not in valid_strategies:
            raise ValueError(
                "combine_strategy must be one of "
                f"['concatenate', 'interleave'], got {combine_strategy}"
            )

        self.combine_strategy = combine_strategy

        super().__init__(
            input_fields=inspect.getfullargspec(load_dataset).args,
            output_fields=fields_to_keep,
        )

    def map(
        self,
        dataset: Any,
        **map_kwargs: Any,
    ) -> Any:
        transformed_dataset = self.transform(dataset)

        self._check_fields_datasets(
            provided_fields=transformed_dataset.features.keys(),
            expected_fields=self.output_fields,
        )

        return transformed_dataset

    # wrapping this in if statement to avoid errors that get raised
    # in case Datasets library is not available.
    if HUGGINGFACE_DATASET_AVAILABLE:

        def transform(
            self, data: Iterable[TransformElementType]
        ) -> Union[Dataset, IterableDataset]:

            datasets_accumulator = []
            for dataset_spec in data:

                # load this specific dataset
                dataset = load_dataset(**dataset_spec)

                # if the user has provided some output fields, we need to check
                # if (1) they are
                if (
                    self.output_fields
                    and getattr(dataset, "features", None) is not None
                ):
                    features = cast(dict, dataset.features)  # pyright: ignore

                    if all(f in features for f in self.output_fields):
                        raise ValueError(
                            f"Dataset {dataset_spec} does not have the "
                            "following fields:  {self.output_fields}"
                        )

                    dataset = dataset.remove_columns(
                        [f for f in features if f not in self.output_fields]
                    )

                datasets_accumulator.append(dataset)

            if len(datasets_accumulator) == 1:
                return datasets_accumulator[0]
            elif self.combine_strategy == "concatenate":
                return concatenate_datasets(datasets_accumulator)
            elif self.combine_strategy == "interleave":
                return interleave_datasets(datasets_accumulator)
            else:
                raise RuntimeError(
                    "This should never happen. Please report this bug."
                )


class CsvLoaderMapper(BatchedBaseMapper):
    def __init__(
        self,
        paths_field: str,
        headers: Optional[List[str]] = None,
        encoding: str = "utf-8",
        **dict_reader_args: Any,
    ) -> None:
        self.paths_field = paths_field
        self.headers = headers
        self.encoding = encoding
        self.dict_reader_args = dict_reader_args
        super().__init__(input_fields=[paths_field], output_fields=headers)

    def transform(
        self, data: Iterable[TransformElementType]
    ) -> Iterable[TransformElementType]:

        for row in data:
            paths = row[self.paths_field]
            if not isinstance(paths, abc.Sequence) or isinstance(paths, str):
                paths = [paths]

            for path in paths:
                with open(path, mode="rt", encoding=self.encoding) as f:
                    if self.headers:
                        yield from DictReader(
                            f, fieldnames=self.headers, **self.dict_reader_args
                        )
                    else:
                        yield from DictReader(f, **self.dict_reader_args)


class JsonlLoaderMapper(BatchedBaseMapper):
    def __init__(
        self,
        paths_field: str,
        encoding: str = "utf-8",
    ) -> None:
        self.paths_field = paths_field
        self.encoding = encoding
        super().__init__(input_fields=[paths_field])

    def transform(
        self, data: Iterable[TransformElementType]
    ) -> Iterable[TransformElementType]:

        for row in data:
            paths = row[self.paths_field]
            if not isinstance(paths, abc.Sequence) or isinstance(paths, str):
                paths = [paths]

            for path in paths:
                with open(path, mode="rt", encoding=self.encoding) as f:
                    for ln in f:
                        yield json.loads(ln)
