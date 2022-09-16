from typing import Union
from unittest import TestCase

from necessary import necessary

with necessary("datasets"):
    from datasets.arrow_dataset import Dataset

from smashed.mappers.types import BinarizerMapper, LookupMapper


class TestBinarizer(TestCase):
    def _run_tests(self, dataset: Union[list, Dataset]):
        mapper = BinarizerMapper(field="a", threshold=0.7) >> BinarizerMapper(
            field="b", threshold=0.7
        )

        mapped_dataset = mapper.map(dataset)

        self.assertEqual(mapped_dataset[0]["a"], [0, 0, 1])
        self.assertEqual(mapped_dataset[0]["b"], 1)

    def test_binarizer_list_of_dict(self):
        dataset = [{"a": [0.3, 0.4, 0.8], "b": 0.9}]
        self._run_tests(dataset)

    def test_binarizer_hf_dataset(self):
        dataset = Dataset.from_dict({"a": [[0.3, 0.4, 0.8]], "b": [0.9]})
        self._run_tests(dataset)

    def test_lookup_mapper(self):
        dataset = [
            {"menu": ["apple", "pie"]},
            {"menu": ["key lime", "pie"]},
            {"menu": ["fudge", "pie"]},
            {"menu": []},
        ]
        mapper = LookupMapper(
            field_name="menu",
            lookup_table={
                "apple": "fruit",
                "key lime": "fruit",
                "pie": "dessert",
                "fudge": "chocolate",
            },
        )
        mapped_dataset = mapper.map(dataset)
        self.assertEqual(mapped_dataset[0]["menu"], ["fruit", "dessert"])
        self.assertEqual(mapped_dataset[1]["menu"], ["fruit", "dessert"])
        self.assertEqual(mapped_dataset[2]["menu"], ["chocolate", "dessert"])
        self.assertEqual(mapped_dataset[3]["menu"], [])
