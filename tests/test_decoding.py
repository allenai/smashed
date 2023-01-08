import unittest

from transformers.models.auto import AutoTokenizer

from smashed.mappers.decoding import DecodingMapper
from smashed.mappers.tokenize import TokenizerMapper


class TestDecoding(unittest.TestCase):
    def setUp(self) -> None:
        self.bert_tok = AutoTokenizer.from_pretrained("bert-base-cased")
        self.gpt2_tok = AutoTokenizer.from_pretrained("gpt2")

    def test_decoding_mapper(self):
        dataset = [
            {
                "source": "Translate english to french : this is a test",
                "target": "c'est un test",
            },
            {
                "source": "Translate english to german : this is another test",
                "target": "Das ist ein anderer test",
            },
            {
                "source": "Translate english to italian : tests are important",
                "target": "I test sono importanti",
            },
        ]

        for tokenizer in [self.bert_tok, self.gpt2_tok]:
            mapper = (
                TokenizerMapper(
                    tokenizer=tokenizer,
                    input_field="source",
                    add_special_tokens=False,
                    return_attention_mask=False,
                    output_rename_map={"input_ids": "source"},
                )
                >> TokenizerMapper(
                    tokenizer=tokenizer,
                    input_field="target",
                    add_special_tokens=False,
                    return_attention_mask=False,
                    output_rename_map={"input_ids": "target"},
                )
                >> DecodingMapper(
                    tokenizer=tokenizer,
                    fields=["source", "target"],
                )
            )

            mapped_dataset = mapper.map(dataset)

            for i, d in enumerate(mapped_dataset):
                self.assertEqual(d["source"], dataset[i]["source"])
                self.assertEqual(d["target"], dataset[i]["target"])
