"""

Tests for TokenizeMappers module

@kylel

"""

import unittest

from transformers.models.auto.tokenization_auto import AutoTokenizer

from smashed.mappers.tokenize import TokenizerMapper, ValidUnicodeMapper


class TestValidUnicodeMapper(unittest.TestCase):
    """Test ValidUnicodeMapper"""

    def test_map(self):
        """Test that ValidUnicodeMapper correctly performs the replacement"""
        mapper = ValidUnicodeMapper(
            input_fields=["tokens"],
            unicode_categories=[
                "Cc",
                "Cf",
                "Co",
                "Cs",
                "Mn",
                "Zl",
                "Zp",
                "Zs",
            ],
            replace_token="[UNK]",
        )
        dataset = [
            {
                "tokens": [
                    "This",
                    "example",
                    "has",
                    "bad",
                    "\uf02a",
                    "\uf02a\u00ad",
                    "Modalities\uf02a",
                ]
            }
        ]
        new_dataset: list = mapper.map(dataset)  # type: ignore
        self.assertListEqual(
            new_dataset,
            [
                {
                    "tokens": [
                        "This",
                        "example",
                        "has",
                        "bad",
                        "[UNK]",
                        "[UNK]",
                        "Modalities\uf02a",
                    ]
                }
            ],
        )


class TestTokenizerMapper(unittest.TestCase):
    """Test TokenizationMapper"""

    def setUp(self):
        """define tokenizer for all tests"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            "allenai/scibert_scivocab_uncased"
        )

    def test_map(self):
        """Test that TokenizationMapper returns the right fields
        and is doing something right"""
        mapper = TokenizerMapper(input_field="text", tokenizer=self.tokenizer)
        dataset = [
            {
                "text": [
                    "This is a sentence.",
                    "This is two sentences. Here is the second one.",
                ]
            },
            {
                "text": [
                    "This is a separate instance.",
                    "This",
                    "is",
                    "some",
                    "tokens",
                    ".",
                ]
            },
        ]
        new_dataset = mapper.map(dataset)

        # same num dicts
        self.assertEqual(len(dataset), len(new_dataset))
        # each element is dict
        self.assertIsInstance(new_dataset[0], dict)

        # dict has keys
        self.assertTrue("input_ids" in new_dataset[0])

        # attention mask is correct
        self.assertTrue("attention_mask" in new_dataset[0])

        # mask same dimension as inputs
        self.assertEqual(
            len(new_dataset[0]["attention_mask"]),
            len(new_dataset[0]["input_ids"]),
        )

        # decoding works as expected
        self.assertEqual(
            self.tokenizer.decode(new_dataset[0]["input_ids"][0]),
            "[CLS] this is a sentence. [SEP]",
        )
        self.assertEqual(
            self.tokenizer.decode(new_dataset[0]["input_ids"][1]),
            "[CLS] this is two sentences. here is the second one. [SEP]",
        )

    def test_truncation_max_length(self):
        """Test that truncation in Tokenization works as intended"""
        mapper = TokenizerMapper(
            input_field="text",
            tokenizer=self.tokenizer,
            truncation=True,
            max_length=10,
        )
        dataset = [
            {
                "text": [
                    "This is an instance that will be truncated "
                    "because it is longer than ten word pieces.",
                    "This is the subsequent unit in this instance "
                    "that will be separately truncated.",
                ]
            },
            {"text": ["This is the next instance."]},
        ]

        new_dataset = mapper.map(dataset)
        assert len(dataset) == len(new_dataset)  # same num dicts
        assert (
            self.tokenizer.decode(new_dataset[0]["input_ids"][0])
            == "[CLS] this is an instance that will be truncated [SEP]"
        )
        assert (
            self.tokenizer.decode(new_dataset[0]["input_ids"][1])
            == "[CLS] this is the subsequent unit in this instance [SEP]"
        )
        assert (
            self.tokenizer.decode(new_dataset[1]["input_ids"][0])
            == "[CLS] this is the next instance. [SEP]"
        )

    def test_overflow(self):
        """
        offsets are start:end char spans into the original text.
        Each wordpiece has its own start/end.
        special tokens like [cls] and [sep] dont have any offsets
        (start == end char)
        """
        mapper = TokenizerMapper(
            input_field="text",
            tokenizer=self.tokenizer,
            truncation=True,
            max_length=10,
            return_overflowing_tokens=True,
        )
        dataset = [
            {
                "text": [
                    "This is an instance that will be truncated because "
                    "it is longer than ten word pieces.",
                    "This is the subsequent unit in this instance that "
                    "will be separately truncated.",
                ]
            },
            {"text": ["This is the next instance."]},
        ]
        new_dataset = mapper.map(dataset)
        assert len(dataset) == len(new_dataset)  # same num dicts
        assert "overflow_to_sample_mapping" in new_dataset[0]
        assert (
            self.tokenizer.decode(new_dataset[0]["input_ids"][0])
            == "[CLS] this is an instance that will be truncated [SEP]"
        )
        assert (
            self.tokenizer.decode(new_dataset[0]["input_ids"][1])
            == "[CLS] because it is longer than ten word pieces [SEP]"
        )
        assert (
            self.tokenizer.decode(new_dataset[0]["input_ids"][2])
            == "[CLS]. [SEP]"
        )
        assert (
            self.tokenizer.decode(new_dataset[0]["input_ids"][3])
            == "[CLS] this is the subsequent unit in this instance [SEP]"
        )
        assert (
            self.tokenizer.decode(new_dataset[0]["input_ids"][4])
            == "[CLS] that will be separately truncated. [SEP]"
        )
        assert (
            self.tokenizer.decode(new_dataset[1]["input_ids"][0])
            == "[CLS] this is the next instance. [SEP]"
        )

    def test_char_offsets(self):
        """Test return value of character offsets is correct"""
        mapper = TokenizerMapper(
            input_field="text",
            tokenizer=self.tokenizer,
            return_offsets_mapping=True,
        )
        dataset = [{"text": ["This is a Pterodactyl."]}]
        new_dataset = mapper.map(dataset)
        assert [
            dataset[0]["text"][0][start:end]
            for start, end in new_dataset[0]["offset_mapping"][0]
        ] == ["", "This", "is", "a", "Pt", "ero", "da", "ct", "yl", ".", ""]

    def test_split_into_words(self):
        """
        We test 3 main functionalities:

        In part 1:
            compare this with `test_char_offsets()`. there,
            we see `offset_mapping` has list elements,
            but here, it's collapsed into one field

        In part 2:
            when `is_split_into_words=False`, `input_ids`
            keeps elements in the List[str] separate. that is,
            tokenizes each separately. When `is_split_into_words=True`,
            `input_ids` collapses the List[str] into a single str for
            tokenization. to see difference, compare with the test
            results from `test_map()`

        In part 3:
            when we set `return_overflowing_tokens=True`, we gain
            the list elements of `input_ids` again.
            what happens here is, the list elements are stitched
            together due to `is_split_into_words=True`,
            then the overflow logic is applied to create more instances.
            key difference versus what we see in `test_overflow()`
            is this no longer adheres to boundaries between List[str]
            elements within an instance (dict). Specifically, the boundary
            between `...ten word pieces.` and `This is the subsequent...`
            is gone now.
        """
        mapper = TokenizerMapper(
            input_field="text",
            tokenizer=self.tokenizer,
            truncation=True,
            max_length=10,
            return_offsets_mapping=True,
            is_split_into_words=True,
        )

        dataset = [{"text": ["This is a Pterodactyl."]}]
        new_dataset = mapper.map(dataset)
        assert [
            dataset[0]["text"][0][start:end]
            for start, end in new_dataset[0]["offset_mapping"]
        ] == ["", "This", "is", "a", "Pt", "ero", "da", "ct", "yl", ""]

        # now try with a larger dataset to test things like
        # truncation and all that
        dataset = [
            {
                "text": [
                    "This is a sentence.",
                    "This is two sentences. Here is the second one.",
                ]
            },
            {
                "text": [
                    "This is a separate instance.",
                    "This",
                    "is",
                    "some",
                    "tokens",
                    ".",
                ]
            },
        ]
        new_dataset = mapper.map(dataset)
        assert (
            self.tokenizer.decode(new_dataset[0]["input_ids"])
            == "[CLS] this is a sentence. this is two [SEP]"
        )
        assert (
            self.tokenizer.decode(new_dataset[1]["input_ids"])
            == "[CLS] this is a separate instance. this is [SEP]"
        )

        # compare with `test_overflow()`
        mapper = TokenizerMapper(
            input_field="text",
            tokenizer=self.tokenizer,
            truncation=True,
            max_length=10,
            return_offsets_mapping=True,
            is_split_into_words=True,
            return_overflowing_tokens=True,
        )
        dataset = [
            {
                "text": [
                    "This is an instance that will be truncated because "
                    "it is longer than ten word pieces.",
                    "This is the subsequent unit in this instance that "
                    "will be separately truncated.",
                ]
            },
            {"text": ["This is the next instance."]},
        ]
        new_dataset = mapper.map(dataset)
        assert (
            self.tokenizer.decode(new_dataset[0]["input_ids"][0])
            == "[CLS] this is an instance that will be truncated [SEP]"
        )
        assert (
            self.tokenizer.decode(new_dataset[0]["input_ids"][1])
            == "[CLS] because it is longer than ten word pieces [SEP]"
        )
        assert (
            self.tokenizer.decode(new_dataset[0]["input_ids"][2])
            == "[CLS]. this is the subsequent unit in this [SEP]"
        )
        assert (
            self.tokenizer.decode(new_dataset[0]["input_ids"][3])
            == "[CLS] instance that will be separately truncated. [SEP]"
        )
        assert (
            self.tokenizer.decode(new_dataset[1]["input_ids"][0])
            == "[CLS] this is the next instance. [SEP]"
        )

    def test_return_words(self):
        """
        there are 2 primary functionalities we need to check here
        first, word pieces are correctly mapped. see how Pterodactyl,
        which is split into 5 wordpieces, is correctly mapped back to
        its original word second, despite truncation & returning overflow
        causing there to be additional new sequences, that we can still
        map back to the original word in the sequence. See how tokens
        in sequence [1] and [2] can still get mapped back to the original word.
        """
        mapper = TokenizerMapper(
            input_field="text",
            tokenizer=self.tokenizer,
            truncation=True,
            max_length=10,
            return_offsets_mapping=True,
            is_split_into_words=True,
            return_overflowing_tokens=True,
            return_word_ids=True,
            return_words=True,
        )
        dataset = [
            {
                "text": [
                    "This",
                    "is",
                    "a",
                    "Pterodactyl",
                    "that",
                    "will",
                    "be",
                    "truncated",
                    "because",
                    "it",
                    "is",
                    "longer",
                    "than",
                    "ten",
                    "word",
                    "pieces",
                    ".",
                    "This",
                    "is",
                    "the",
                    "subsequent",
                    "Pterodactyl",
                    "in",
                    "this",
                    "instance",
                    "that",
                    "will",
                    "be",
                    "separately",
                    "truncated",
                    ".",
                ]
            }
        ]
        new_dataset = mapper.map(dataset)
        assert "words" in new_dataset[0]
        assert "word_ids" in new_dataset[0]
        self.assertEqual(
            new_dataset[0]["words"][0],
            [
                None,
                "This",
                "is",
                "a",
                "Pterodactyl",
                "Pterodactyl",
                "Pterodactyl",
                "Pterodactyl",
                "Pterodactyl",
                None,
            ],
        )
        self.assertEqual(
            new_dataset[0]["words"][1],
            [
                None,
                "that",
                "will",
                "be",
                "truncated",
                "because",
                "it",
                "is",
                "longer",
                None,
            ],
        )
        self.assertEqual(
            new_dataset[0]["words"][2],
            [
                None,
                "than",
                "ten",
                "word",
                "pieces",
                ".",
                "This",
                "is",
                "the",
                None,
            ],
        )

    def test_prefix(self):
        mapper = TokenizerMapper(
            input_field="text",
            tokenizer=self.tokenizer,
            return_attention_mask=False,
            output_prefix="test",
        )

        dataset = [
            {"text": "This is a sentence."},
        ]

        new_dataset = mapper.map(dataset)
        self.assertEqual("test_input_ids" in new_dataset[0], True)
        self.assertEqual(
            new_dataset[0]["test_input_ids"],
            [102, 238, 165, 106, 8517, 205, 103],
        )

    def test_rename(self):
        mapper = TokenizerMapper(
            input_field="text",
            tokenizer=self.tokenizer,
            return_attention_mask=True,
            output_rename_map={"input_ids": "foo", "attention_mask": "bar"},
        )

        dataset = [
            {"text": "This is a sentence."},
        ]

        new_dataset = mapper.map(dataset)
        self.assertTrue("foo" in new_dataset[0])
        self.assertTrue("bar" in new_dataset[0])
        self.assertFalse("input_ids" in new_dataset[0])
        self.assertFalse("attention_mask" in new_dataset[0])

        with self.assertRaises(ValueError):
            mapper = TokenizerMapper(
                input_field="text",
                tokenizer=self.tokenizer,
                return_attention_mask=True,
                return_token_type_ids=True,
                output_rename_map={
                    "input_ids": "foo",
                    "attention_mask": "bar",
                },
            )
            mapper.map(dataset)
