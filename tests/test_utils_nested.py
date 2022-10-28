from unittest import TestCase

from smashed.utils.nested import Nested


class TestUtilsNested(TestCase):
    def test_with_lists(self):

        data = {
            "a": {
                "b": [
                    {"c": {"d": 1, "f": 3}},
                    {"c": {"d": 2, "g.h": [1, 3, 5]}},
                ],
                "k": 3,
            },
            "h": 4,
        }

        path = Nested.from_str("a.b.[c.d]")
        self.assertEqual(str(path), "a.b.[c.d]")

        out = path.edit(data, lambda x: x + 1, inplace=False)
        self.assertEqual(out["a"]["b"][0]["c"], {"d": 2, "f": 3})
        self.assertEqual(out["a"]["b"][1]["c"], {"d": 3, "g.h": [1, 3, 5]})
        self.assertEqual(out["a"]["k"], 3)
        self.assertEqual(out["h"], 4)

        out = path.copy(data)
        self.assertEqual(out["a"]["b"], [{"c": {"d": 1}}, {"c": {"d": 2}}])
        self.assertFalse("f" in out["a"]["b"][0]["c"])
        self.assertFalse("g.h" in out["a"]["b"][1]["c"])
        self.assertFalse("k" in out["a"])
        self.assertFalse("h" in out)

        out = path.select(data)
        self.assertEqual(out, [1, 2])

    def test_with_list_single(self):
        data = {
            "a": {
                "b": [
                    {"c": {"d": 1, "f": 3}},
                    {"c": {"d": 2, "g.h": [1, 3, 5]}},
                ],
                "k": 3,
            },
            "h": 4,
        }

        path = Nested.from_str("a.b.0.c")
        self.assertEqual(str(path), "a.b.0.c")

        out = path.edit(
            data, lambda x: {k: v + 1 for k, v in x.items()}, inplace=False
        )
        self.assertEqual(out["a"]["b"][0]["c"], {"d": 2, "f": 4})
        self.assertEqual(out["a"]["b"][1]["c"], {"d": 2, "g.h": [1, 3, 5]})

        out = path.copy(data)
        self.assertEqual(out["a"]["b"], [{"c": {"d": 1, "f": 3}}])

        out = path.select(data)
        self.assertEqual(out, {"d": 1, "f": 3})

    def test_quote_in_keys(self):
        data = {
            "a": {
                "b": [
                    {"c": {"d": 1, "f": 3}},
                    {"c": {"d": 2, "g.h": [1, 3, 5]}},
                ],
                "k": 3,
            },
            "h": 4,
        }
        path = Nested.from_str("a.b.(c.'g.h')")
        self.assertEqual(str(path), 'a.b.[c."g.h"]')

        path.edit(
            data, lambda x: [v + 1 for v in x], inplace=True, missing=None
        )
        self.assertNotIn("g.h", data["a"]["b"][0]["c"])
        self.assertEqual(data["a"]["b"][1]["c"]["g.h"], [2, 4, 6])

        out = path.copy(data, missing=None)
        self.assertEqual(out["a"]["b"][0]["c"], {"g.h": None})
        self.assertEqual(out["a"]["b"][1]["c"], {"g.h": [2, 4, 6]})

        out = path.select(data, missing=None)
        self.assertEqual(out, [None, 2, 4, 6])
