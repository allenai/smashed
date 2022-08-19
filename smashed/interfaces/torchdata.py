from typing import Any, Dict, TypeVar, Union
from collections import abc

from ..base.mapper import DatasetInterfaceMapper
from ..mappers import fields, multiseq, shape, tokenize
from ..mappers.contrib import sse
from ..utils import requires
from ..base.types import DatasetType, TorchDataDatasetType

requires("torch", "1.12.0")
requires("torchdata", "0.4.0")

from torchdata.datapipes.map import SequenceWrapper
from torchdata.datapipes.iter import IterableWrapper


def Dataset(dataset: DatasetType) -> TorchDataDatasetType:
    if isinstance(dataset, abc.Sequence):
        return SequenceWrapper(dataset)
    else:
        return IterableWrapper(dataset)


class TorchDataDatasetsInterfaceMapper(DatasetInterfaceMapper):

    def get_dataset_fields(
        self: "TorchDataDatasetsInterfaceMapper",
        dataset: TorchDataDatasetType
    ) -> Union[Iterable[str], None]:
        ...

    def check_dataset_fields(
        self: "TorchDataDatasetsInterfaceMapper",
        provided_fields: Union[Iterable[str], None],
        expected_fields: Sequence[str],
    ):
        ...

    def map(
        self: "TorchDataDatasetsInterfaceMapper",
        dataset: TorchDataDatasetType,
        **_: Any,
    ) -> TorchDataDatasetType:
        self.check_dataset_fields(
            provided_fields=self.get_dataset_fields(dataset),
            expected_fields=self.input_fields,
        )
        dataset = self.transform(dataset)
        return dataset
