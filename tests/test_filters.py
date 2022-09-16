import unittest

from smashed.mappers.filters import FilterMapper


class TestFilterMapper(unittest.TestCase):
    def test_filter(self):
        mapper = FilterMapper(field_name="a", operator=">=", value=5)
        data = [{"a": 5}, {"a": 4}, {"a": 6}]
        self.assertEqual(list(mapper.map(data)), [{"a": 5}, {"a": 6}])

    def test_filter_list(self):
        data = [{"a": [5]}, {"a": [4]}, {"a": [6]}]
        mapper = FilterMapper(field_name="a", operator="==", value=5)
        self.assertEqual(list(mapper.map(data)), [{"a": [5]}])
