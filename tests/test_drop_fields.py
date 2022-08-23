"""

Test functionality to drop fields from a dataset.

@lucas

"""

import unittest

from transformers.models.auto.tokenization_auto import AutoTokenizer

from smashed.mappers.tokenize import TokenizerMapper


class DropFieldTest(unittest.TestCase):
    def test_dropping_fields(self):
        dataset = [
            {"text": "Hello, world!", "label": 1},
            {"text": "Bye suckers!", "label": 0},
        ]

        mapper = TokenizerMapper(
            tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"),
            input_field="text",
            return_attention_mask=False,
        )
        processed_dataset = mapper.map(dataset, remove_columns=True)
        self.assertEqual(processed_dataset[0].keys(), {"input_ids"})

        processed_dataset = mapper.map(dataset, remove_columns=False)
        self.assertEqual(
            processed_dataset[0].keys(), {"text", "input_ids", "label"}
        )
