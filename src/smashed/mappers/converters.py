from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, TypeVar, Union

import torch
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


class Python2TorchMapper(SingleBaseMapper):

    __slots__ = ["field_cast_map", "device"]
    field_cast_map: Dict[str, torch.dtype]
    device: Union[torch.device, None]

    def __init__(
        self: "Python2TorchMapper",
        field_cast_map: Optional[Mapping[str, Union[str, torch.dtype]]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """Mapper that converts Python types to Torch types. It can optionally
        cast the values of a field to a specific type, and move to a specific
        device.

        Args:
            field_cast_map (Mapping[str, Union[str, torch.dtype]], optional):
                Mapping from field names to the types to cast the values to.
                Defaults to None, which means no casting occurs.
            device (Union[torch.device, str], optional): Device to move the
                tensors to. Defaults to None, which means no moving occurs.
        """
        self.device = torch.device(device) if device else None

        self.field_cast_map = {
            field_name: self._get_dtype(field_type)
            for field_name, field_type in (field_cast_map or {}).items()
        }
        super().__init__(
            input_fields=list(self.field_cast_map.keys()),
            output_fields=list(self.field_cast_map.keys()),
        )

    @staticmethod
    def _get_dtype(dtype: Any) -> torch.dtype:

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype, None)
            if dtype is None:
                raise ValueError(f"Unknown dtype {dtype}")

        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"{dtype} is not a torch dtype")

        return dtype

    def transform(self, data: TransformElementType) -> TransformElementType:
        return {
            # .to(type) and .to(device) will return the original tensor if
            # the type and device are the same/None; this is because
            # for any torch.Tensor t, hash(t) == hash(t.to(None).to(None))
            field_name: torch.tensor(field_value)
            .to(self.field_cast_map.get(field_name, None))
            .to(self.device)
            for field_name, field_value in data.items()
        }

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

    if HUGGINGFACE_DATASET_AVAILABLE:

        @map.add_interface(dataset=(Dataset, IterableDataset))
        def _map_huggingface_dataset(
            self,
            dataset: HuggingFaceDataset,
            **map_kwargs: Any,
        ) -> HuggingFaceDataset:
            return dataset.with_format("torch")


class Torch2PythonMapper(SingleBaseMapper):
    def __init__(self: "Torch2PythonMapper") -> None:
        """Mapper that converts Torch types to Python types. It relies on
        function `.tolist()` to convert the tensor to a list. It additionally
        moves the tensor to the CPU before converting it.
        """
        super().__init__()

    def transform(  # type: ignore
        self: "Torch2PythonMapper", data: Dict[str, torch.Tensor]
    ) -> TransformElementType:
        return {
            field_name: field_value.cpu().tolist()
            for field_name, field_value in data.items()
        }

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

    if HUGGINGFACE_DATASET_AVAILABLE:

        @map.add_interface(dataset=(Dataset, IterableDataset))
        def _map_huggingface_dataset(
            self,
            dataset: HuggingFaceDataset,
            **map_kwargs: Any,
        ) -> HuggingFaceDataset:
            return dataset.with_format(None)
