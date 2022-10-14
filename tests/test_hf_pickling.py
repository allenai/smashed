import pickle
import unittest

from necessary import necessary

from smashed.mappers.debug import MockMapper

with necessary(("datasets", "dill")):
    import dill
    from datasets.fingerprint import Hasher


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
