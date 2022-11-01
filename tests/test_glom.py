import unittest

from necessary import necessary

from smashed.mappers.glom import GlomMapper

with necessary("datasets"):
    from datasets.arrow_dataset import Dataset
    from pandas import DataFrame


class TestGlom(unittest.TestCase):
    @property
    def dataset(self):
        return [
            {
                "id": "56be4db0acb8001400a502ec",
                "title": "Super_Bowl_50",
                "context": (
                    "Super Bowl 50 was an American football game to determine"
                    " the champion of the National Football League (NFL) for "
                    "the 2015 season. The American Football Conference (AFC)"
                    " champion Denver Broncos defeated the National Football "
                    "Conference (NFC) champion Carolina Panthers 24–10 to "
                    "earn their third Super Bowl title."
                ),
                "question": (
                    "Which NFL team represented the AFC at Super Bowl 50?"
                ),
                "answers": {
                    "text": [
                        "Denver Broncos",
                        "Denver Broncos",
                        "Denver Broncos",
                    ],
                    "answer_start": [177, 177, 177],
                },
            },
            {
                "id": "572763a8708984140094dcda",
                "title": "American_Broadcasting_Company",
                "context": (
                    "In 1990, Thomas S. Murphy delegated his position as "
                    "president to Daniel B. Burke while remaining ABC's "
                    "chairman and CEO. Capital Cities/ABC reported revenues "
                    "of $465 million. Now at a strong second place, the "
                    "network entered the 1990s with additional "
                    "family-friendly hits including America's Funniest Home "
                    "Videos, Step by Step, Hangin' with Mr. Cooper, Boy Meets"
                    " World and Perfect Strangers spinoff Family Matters, as "
                    "well as series such as Doogie Howser, M.D., Life Goes On,"
                    " cult favorite Twin Peaks and The Commish."
                ),
                "question": (
                    "What position at ABC did Thomas Murphy stay on for after"
                    " stepping down as president?"
                ),
                "answers": {
                    "text": [
                        "chairman and CEO",
                        "chairman and CEO",
                        "chairman",
                    ],
                    "answer_start": [103, 103, 103],
                },
            },
            {
                "id": "5725e152271a42140099d2d1",
                "title": "Apollo_program",
                "context": (
                    "Apollo 5 (AS-204) was the first unmanned test flight of "
                    "LM in Earth orbit, launched from pad 37 on January 22, "
                    "1968, by the Saturn IB that would have been used for "
                    "Apollo 1. The LM engines were successfully test-fired and"
                    "restarted, despite a computer programming error which cut"
                    " short the first descent stage firing. The ascent engine"
                    " was fired in abort mode, known as a 'fire-in-the-hole' "
                    "test, where it was lit simultaneously with jettison of "
                    "the descent stage. Although Grumman wanted a second "
                    "unmanned test, George Low decided the next LM flight "
                    "would be manned."
                ),
                "question": (
                    "What was the nickname for the test where, during abort "
                    "mode, the ascent engine was started and fired?"
                ),
                "answers": {
                    "text": [
                        "'fire-in-the-hole'",
                        "fire-in-the-hole",
                        "fire-in-the-hole",
                        "fire-in-the-hole",
                        "'fire-in-the-hole'",
                    ],
                    "answer_start": [372, 373, 373, 373, 372],
                },
            },
        ]

    def test_glom_mapper_list_of_dicts(self):
        spec = ("answers", "text", tuple())
        gm = GlomMapper(spec_fields={"answers": spec})
        out = gm.map(self.dataset)

        self.assertEqual(len(out), 3)
        self.assertEqual(
            out[0]["answers"],
            ["Denver Broncos", "Denver Broncos", "Denver Broncos"],
        )
        self.assertEqual(
            out[1]["answers"],
            ["chairman and CEO", "chairman and CEO", "chairman"],
        )
        self.assertEqual(
            out[2]["answers"],
            [
                "'fire-in-the-hole'",
                "fire-in-the-hole",
                "fire-in-the-hole",
                "fire-in-the-hole",
                "'fire-in-the-hole'",
            ],
        )

    def test_glom_mapper_huggingface_dataset(self):
        dataset = Dataset.from_pandas(DataFrame.from_records(self.dataset))
        gm = GlomMapper(spec_fields={"answers": ("answers", "text", tuple())})
        out = gm.map(dataset)

        self.assertEqual(len(out), 3)
        self.assertEqual(
            out[0]["answers"],
            ["Denver Broncos", "Denver Broncos", "Denver Broncos"],
        )
        self.assertEqual(
            out[1]["answers"],
            ["chairman and CEO", "chairman and CEO", "chairman"],
        )
        self.assertEqual(
            out[2]["answers"],
            [
                "'fire-in-the-hole'",
                "fire-in-the-hole",
                "fire-in-the-hole",
                "fire-in-the-hole",
                "'fire-in-the-hole'",
            ],
        )
