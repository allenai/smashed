import unittest
from copy import deepcopy
from functools import partial

from datasets.arrow_dataset import Dataset

try:
    from datasets.formatting.formatting import LazyBatch
except ImportError:
    # pre datasets 2.8.0
    from datasets.arrow_dataset import Batch as LazyBatch  # pyright: ignore

from smashed.mappers.debug import MockMapper


class TestBatchInterface(unittest.TestCase):
    def test_batch(self, remove_columns: bool = False):
        mapper = MockMapper(1, output_fields=["a"])

        data = Dataset.from_list([{"a": i, "b": i ** 2} for i in range(100)])

        def _batch_fn(data: LazyBatch, mapper: MockMapper) -> LazyBatch:
            return mapper.map(deepcopy(data), remove_columns=remove_columns)

        fn = partial(_batch_fn, mapper=mapper)

        out = data.map(
            fn,
            batched=True,
            batch_size=10,
            remove_columns=["b"] if remove_columns else None,
        )

        self.assertEqual(len(out), 100)
        if remove_columns:
            self.assertEqual([f for f in out.features.keys()], ["a"])
        else:
            self.assertEqual([f for f in out.features.keys()], ["a", "b"])

        self.assertEqual(out["a"], [i + 1 for i in range(100)])

    def test_batch_remove_columns(self):
        self.test_batch(remove_columns=True)
