import json
from collections import abc
from csv import DictReader
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, TypeVar, Union

from necessary import necessary
from trouting import trouting

from ..base.mappers import BatchedBaseMapper
from ..base.types import TransformElementType

with necessary("datasets", soft=True) as HUGGINGFACE_DATASET_AVAILABLE:
    if HUGGINGFACE_DATASET_AVAILABLE or TYPE_CHECKING:
        from datasets.arrow_dataset import Dataset
        from datasets.dataset_dict import DatasetDict, IterableDatasetDict
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
        **load_datasets_args: Any,
    ):
        if "path" not in load_datasets_args:
            raise ValueError(
                "datasets.load_dataset requires a path to a dataset."
            )

        self.load_datasets_args = load_datasets_args

    @trouting
    def map(  # type: ignore
        self,
        dataset: Any,
        **map_kwargs: Any,
    ) -> Any:
        # we need this map to be able to add the new interface below
        # and handle types for which we don't have a new interface but our
        # parent class has one
        super().map(dataset, **map_kwargs)

    @map.add_interface(dataset=type(None))
    def map_none(  # type: ignore
        self,
        dataset: None,
        **map_kwargs: Any,
    ) -> Any:
        # this loader
        return self.transform([])

    if HUGGINGFACE_DATASET_AVAILABLE:

        def transform(
            self, data: Iterable[TransformElementType]
        ) -> Union[Dataset, IterableDataset]:
            dataset = load_dataset(**self.load_datasets_args)

            if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
                raise ValueError(
                    "HuggingFaceDatasetLoader only supports loading a single "
                    f"dataset, but the provided dataset is a {type(dataset)}. "
                    "Please provide a `split` to this mapper."
                )

            return dataset


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
