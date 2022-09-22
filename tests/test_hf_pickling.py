import pickle
import unittest

from necessary import necessary

from smashed.base import SingleBaseMapper

with necessary("datasets"):
    from datasets.fingerprint import Hasher


class MockMapper(SingleBaseMapper):
    """A mock mapper that returns the same data it receives.
    Used for testing."""

    __slots__ = ("v",)

    def __init__(self, v: int = 1):
        self.v = v
        super().__init__()

    def transform(self, data: dict) -> dict:
        return {k: v + self.v for k, v in data.items()}


class TestPickling(unittest.TestCase):
    def test_pickle(self):
        """Test if caching works"""

        # this should not fail
        m = MockMapper() >> MockMapper()
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


if __name__ == "__main__":
    import springs as sp

    sp.configure_logging()
    TestPickling().test_pickle()
