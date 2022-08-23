from typing import TYPE_CHECKING, Callable, Generic, TypeVar, Union

from typing_extensions import Concatenate, ParamSpec

from .types import DatasetType

if TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset
    from datasets.iterable_dataset import IterableDataset

HuggingFaceDatasetType = Union["Dataset", "IterableDataset"]

P = ParamSpec("P")
D = TypeVar("D")
M = TypeVar("M")


class interfaces(Generic[M, D, P]):

    __slots__ = ["base_method", "huggingface_method", "torchdata_method"]

    def __init__(
        self: "interfaces",
        method: Callable[Concatenate[M, DatasetType, P], DatasetType],
    ):
        self.base_method = method

    def huggingface(
        self: "interfaces",
        method: Callable[
            Concatenate[HuggingFaceDatasetType, P], HuggingFaceDatasetType
        ],
    ) -> "interfaces":
        self.huggingface_method = method
        return self

    def torchdata(
        self: "interfaces", method: Callable[Concatenate[M, D, P], D]
    ) -> "interfaces":
        raise NotImplementedError()
        # self.torchdata_method = method
        # return self

    def __call__(self, dataset, *args, **kwargs):
        pass


class Mapper:
    @interfaces
    def foo(self, dataset: DatasetType) -> DatasetType:
        return []
