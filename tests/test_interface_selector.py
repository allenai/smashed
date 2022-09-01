import unittest
from typing import Type, TypeVar

from necessary import necessary

with necessary("torch", soft=True) as HAS_TORCH_FLAG:
    import torch

with necessary("numpy", soft=True) as HAS_NUMPY_FLAG:
    import numpy as np

from smashed.base.interface import interface

T = TypeVar("T")


class Mapper:
    """
    InterfaceMapper is a base class for all interface mappers.
    """

    @interface
    def map(self, value: T) -> Type[T]:
        raise NotImplementedError()

    @map.add_interface(value=list)  # type: ignore
    def map_list(self, value: list) -> Type[list]:
        return list

    @map.add_interface(value=dict)  # type: ignore
    def map_dict(self, value: dict) -> Type[dict]:
        return dict

    if HAS_TORCH_FLAG:

        @map.add_interface(value=torch.Tensor)
        def map_tensor(self, value: torch.Tensor) -> Type[torch.Tensor]:
            return torch.Tensor


class Mapper2(Mapper):
    @interface
    def map(self, value: T) -> Type[T]:
        return super().map(value)

    if HAS_NUMPY_FLAG:

        @map.add_interface(value=np.ndarray)
        def map_ndarray(self, value: np.ndarray) -> Type[np.ndarray]:
            return np.ndarray


class TestInterface(unittest.TestCase):
    def test_base_interface(self):
        mapper: Mapper = Mapper()

        arr = ["a", "b", "c"]
        self.assertEqual(mapper.map(arr), list)

        di = {"a": 1, "b": 2, "c": 3}
        self.assertEqual(mapper.map(di), dict)

        t = torch.Tensor([1, 2, 3, 4])
        self.assertEqual(mapper.map(t), torch.Tensor)

        with self.assertRaises(NotImplementedError):
            mapper.map(1)

        with self.assertRaises(NotImplementedError):
            mapper.map(np.array([1, 2, 3]))

    def test_inherited_interface(self):
        mapper: Mapper2 = Mapper2()

        arr = ["a", "b", "c"]
        self.assertEqual(mapper.map(arr), list)

        arr = np.array([1, 2, 3])
        self.assertEqual(mapper.map(arr), np.ndarray)
