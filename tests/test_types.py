from pathlib import Path
from typing import Union
from unittest import TestCase

from necessary import necessary

with necessary("datasets"):
    from datasets.arrow_dataset import Dataset

from smashed.mappers.types import BinarizerMapper


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
