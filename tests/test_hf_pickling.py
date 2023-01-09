import pickle
import unittest
from uuid import uuid4

from necessary import necessary

from smashed.contrib.squad import ConcatenateContextMapper
from smashed.mappers import (
    EnumerateFieldMapper,
    JinjaMapper,
    TokenizerMapper,
    TruncateMultipleFieldsMapper,
    UnpackingMapper,
)
from smashed.mappers.debug import MockMapper

with necessary(["datasets", "dill"]):
    import dill
    from datasets.arrow_dataset import Dataset
    from datasets.fingerprint import Hasher

with necessary("transformers"):
    from transformers import AutoTokenizer


class TestPickling(unittest.TestCase):
    def test_pickle(self):
        """Test if caching works"""

        # this should not fail
        m = MockMapper(1) >> MockMapper(1)
        m2 = pickle.loads(pickle.dumps(m))
        self.assertEqual(m, m2)

        # the pickled pipeline should yield same results
        dt = [{"a": 1, "b": 2}]
        out1 = m.map(dt)
        out2 = m2.map(dt)
        self.assertEqual(out1, out2)

        # this should not fail if class is picklable
        hasher = Hasher()
        hasher.update(m.transform)
        hasher.hexdigest()

    def test_dill(self):
        """Test if caching works"""

        # this should not fail
        m = MockMapper(1) >> MockMapper(1)
        m2 = dill.loads(dill.dumps(m))
        self.assertEqual(m, m2)

        # the dilled pipeline should yield same results
        dt = [{"a": 1, "b": 2}]
        out1 = m.map(dt)
        out2 = m2.map(dt)
        self.assertEqual(out1, out2)

        # this should not fail if class is dillable
        hasher = Hasher()
        hasher.update(m.transform)
        hasher.hexdigest()

    def test_unpacking_fingerprint(self):
        """Test if fingerprinting works"""
        mp = (
            UnpackingMapper(
                fields_to_unpack=["a", "b"], ignored_behavior="drop"
            )
            >> MockMapper(1)
            >> MockMapper(1)
        )

        dataset = Dataset.from_dict({"a": [[1, 2, 3]], "b": [[4, 5, 6]]})

        hashes = set()
        for _ in range(100):
            processed_dataset = mp.map(dataset)
            hashes.add(processed_dataset._fingerprint)

        self.assertEqual(len(hashes), 1)

    def test_tokenizer_fingerprint(self):
        dataset = Dataset.from_dict(
            {"a": ["hello world", "my name is john doe"]}
        )

        mp = TokenizerMapper(
            tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"),
            input_field="a",
        )

        hashes = set()
        for _ in range(100):
            processed_dataset = mp.map(dataset)
            hashes.add(processed_dataset._fingerprint)

        self.assertEqual(len(hashes), 1)

    def test_truncate_fingerprint(self):
        mp = TruncateMultipleFieldsMapper(
            fields_to_truncate=["a", "b"], max_length=2
        )

        dataset = Dataset.from_dict({"a": [[1, 2, 3]], "b": [[4, 5, 6]]})

        hashes = set()
        for _ in range(100):
            processed_dataset = mp.map(dataset)
            hashes.add(processed_dataset._fingerprint)

        self.assertEqual(len(hashes), 1)

    def test_concat_context(self):
        mp = ConcatenateContextMapper()
        dataset = Dataset.from_dict(
            {
                "context": [
                    ["hello world", "my name is john doe"],
                    ["simple string"],
                ]
            }
        )

        hashes = set()
        for _ in range(100):
            processed_dataset = mp.map(dataset)
            hashes.add(processed_dataset._fingerprint)

        self.assertEqual(len(hashes), 1)

    def test_enumerate(self):
        mp = EnumerateFieldMapper("answers")
        dataset = Dataset.from_dict(
            {"answers": [uuid4().hex for _ in range(20)] * 2}
        )

        hashes = set()
        for _ in range(20):
            processed_dataset = mp.map(dataset)
            hashes.add(processed_dataset._fingerprint)

        self.assertEqual(len(hashes), 1)
        self.assertEqual(
            mp.map(dataset)["answers"],
            [i for i in range(20)] + [i for i in range(20)],
        )

    def test_promptsource(self):
        mp = JinjaMapper(jinja="hello {{world}}")

        dataset = Dataset.from_dict(
            {"world": [uuid4().hex for _ in range(20)] * 2}
        )

        # make sure that serialization works both ways
        mp = dill.loads(dill.dumps(mp))

        hashes = set()
        for _ in range(20):
            processed_dataset = mp.map(dataset)
            hashes.add(processed_dataset._fingerprint)

        self.assertEqual(len(hashes), 1)
