import unittest

from smashed.mappers import UnpackingMapper


class TestUnpackingMapper(unittest.TestCase):
    def test_unpack_single(self):
        mapper = UnpackingMapper()
        dataset = [{"a": [0, 1, 2, 3]}, {"a": [4, 5]}]
        dataset = mapper.map(dataset)

        self.assertEqual(len(dataset), 6)
        for i in range(6):
            self.assertEqual(dataset[i], {"a": i})

    def test_unpack_multiple(self):
        mapper = UnpackingMapper()
        dataset = [
            {"a": [0.1, 1.1, 2.1, 3.1], "b": [0.2, 1.2, 2.2, 3.2]},
            {"a": [4.1, 5.1], "b": [4.2, 5.2]},
        ]
        dataset = mapper.map(dataset)

        self.assertEqual(len(dataset), 6)
        for i in range(6):
            self.assertEqual(
                dataset[i], {"a": float(f"{i}.1"), "b": float(f"{i}.2")}
            )

    def test_unpack_multiple_while_skipping_fields(self):
        mapper = UnpackingMapper(
            fields_to_unpack=["a"], ignored_behavior="drop"
        )
        dataset = [
            {"a": [0, 1, 2, 3], "b": "hello"},
            {"a": [4, 5], "b": "hello"},
        ]

        dropped_dataset = mapper.map(dataset)
        self.assertEqual(len(dropped_dataset), 6)
        for i in range(6):
            self.assertEqual(dropped_dataset[i], {"a": i})
            self.assertFalse("b" in dropped_dataset[i])

        mapper = UnpackingMapper(
            fields_to_unpack=["a"], ignored_behavior="repeat"
        )
        repeated_dataset = mapper.map(dataset)
        self.assertEqual(len(repeated_dataset), 6)
        for i in range(6):
            self.assertEqual(repeated_dataset[i], {"a": i, "b": "hello"})
