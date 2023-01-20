import unittest

from smashed.utils.shape_utils import (
    flatten_with_indices,
    reconstruct_from_indices,
)


class TestFlatten(unittest.TestCase):
    def test_flatten(self):
        li = [
            [0, 1, 2, 3],
            ["4", "5"],
            [6, 7],
            ["8"],
            [9.0, 10.0, 11.0, 12.0, 13.0],
            [],
            [14, 15, 16],
            [17, 18, 19, "20"],
            [21, "22"],
            [""],
            [23, 24, 25, 26, 27, 28, 29, "30"],
        ]

        fl, idx = flatten_with_indices(li)
        new_li = reconstruct_from_indices(fl, idx)

        self.assertEqual(li, new_li)

    def test_deeply_nested(self):
        # a nested 4-deep nested list
        li = [
            [[[0, 1, 2, 3], ["4", "5"]], [[6, 7], ["8"]]],
            [
                [[9.0, 10.0, 11.0, 12.0, 13.0], []],
                [[14, 15, 16], [17, 18, 19, "20"], [21, "22"], [""]],
                [[23, 24, 25, 26, 27, 28, 29, "30"]],
            ],
        ]

        fl, idx = flatten_with_indices(li)
        new_li = reconstruct_from_indices(fl, idx)

        self.assertEqual(li, new_li)

    def test_empty(self):
        li = []
        fl, idx = flatten_with_indices(li)
        new_li = reconstruct_from_indices(fl, idx)

        self.assertEqual(li, new_li)

    def test_already_flat(self):
        li = [0, 1, 2, 3]
        fl, idx = flatten_with_indices(li)
        new_li = reconstruct_from_indices(fl, idx)

        self.assertEqual(li, new_li)

    def test_error_when_mixed(self):
        li = [0, 1, 2, 3, [4, 5, 6]]
        with self.assertRaises(ValueError):
            flatten_with_indices(li)
