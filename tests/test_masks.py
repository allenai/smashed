import unittest

from smashed.mappers import (
    IndicesToMaskMapper,
    MaskToIndicesMapper,
    MaskToRangeMapper,
    RangeToMaskMapper,
)

# [CLS] Joseph Robinette Biden Jr. is the US president. [SEP]
SAMPLE = {
    "input_ids": [
        101,
        3312,
        5863,
        7585,
        7226,
        2368,
        3781,
        1012,
        2003,
        1996,
        2149,
        2343,
        1012,
        102,
    ],
    "attention_mask": [1] * 14,
}


class TestMasks(unittest.TestCase):
    def test_indices_to_mask(self):
        mapper = IndicesToMaskMapper(
            mask_field_name="people_mask",
            locations_field_name="people",
            reference_field_name="input_ids",
        )

        dataset = [
            {**SAMPLE, "people": [1, 2, 3, 4, 5, 6, 7, 8], "orgs": [10]}
        ]

        sample, *_ = mapper.map(dataset)

        self.assertIn("people_mask", sample)
        self.assertEqual(len(sample["people_mask"]), 14)
        self.assertEqual(
            sample["people_mask"], [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        )

    def test_range_to_mask(self):
        mapper = RangeToMaskMapper(
            mask_field_name="people_mask",
            locations_field_name="people",
            reference_field_name="input_ids",
        ) >> RangeToMaskMapper(
            mask_field_name="orgs_mask",
            locations_field_name="orgs",
            reference_field_name="input_ids",
        )

        dataset = [{**SAMPLE, "people": [[1, 9]], "orgs": [[10, 11]]}]

        sample, *_ = mapper.map(dataset)

        self.assertIn("people_mask", sample)
        self.assertEqual(len(sample["people_mask"]), 14)
        self.assertEqual(
            sample["people_mask"], [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        )

        self.assertIn("orgs_mask", sample)
        self.assertEqual(len(sample["orgs_mask"]), 14)
        self.assertEqual(
            sample["orgs_mask"], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        )

    def test_mask_to_indices(self):
        mapper = MaskToIndicesMapper(
            mask_field_name="people_mask",
            locations_field_name="people",
        )

        dataset = [
            {
                **SAMPLE,
                "people_mask": [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            }
        ]

        sample, *_ = mapper.map(dataset)

        self.assertIn("people", sample)
        self.assertEqual(sample["people"], [1, 2, 3, 4, 5, 6, 7, 8])

    def test_mask_to_range(self):
        mapper = MaskToRangeMapper(
            mask_field_name="people_mask",
            locations_field_name="people",
        ) >> MaskToRangeMapper(
            mask_field_name="orgs_mask",
            locations_field_name="orgs",
        )

        dataset = [
            {
                **SAMPLE,
                "people_mask": [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                "orgs_mask": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            }
        ]

        sample, *_ = mapper.map(dataset)

        self.assertIn("people", sample)
        self.assertEqual(sample["people"], [[1, 9]])

        self.assertIn("orgs", sample)
        self.assertEqual(sample["orgs"], [[10, 11]])
