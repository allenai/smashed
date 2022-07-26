import unittest

import transformers
from smashed.mappers.tokenize import TokenizerMapper, ValidUnicodeMapper
from smashed.interfaces.simple import Dataset, TokenizerMapper, ValidUnicodeMapper

class TestValidUnicodeMapper(unittest.TestCase):
    def test_map(self):
        mapper = ValidUnicodeMapper(
            input_fields=['tokens'],
            unicode_categories=["Cc", "Cf", "Co", "Cs", "Mn", "Zl", "Zp", "Zs"],
            replace_token='[UNK]'
        )
        dataset = Dataset([
            {
                'tokens': ['This', 'example', 'has', 'bad', "\uf02a", "\uf02a\u00ad", "Modalities\uf02a"]
            }
        ])
        new_dataset = mapper.map(dataset)
        self.assertListEqual(new_dataset, [
            {'tokens': ['This', 'example', 'has', 'bad', '[UNK]', '[UNK]', 'Modalities\uf02a']}
        ])


class TestTokenizerMapper(unittest.TestCase):
    def setUp(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            'allenai/scibert_scivocab_uncased'
        )

    def test_map(self):
        mapper = TokenizerMapper(
            input_field='text',
            tokenizer=self.tokenizer
        )
        dataset = Dataset([
            {
                'text': [
                    'This is a sentence.',
                    'This is two sentences. Here is the second one.'
                ]
            },
            {
                'text': [
                    'This is a separate instance.',
                    'This',
                    'is',
                    'some',
                    'tokens',
                    '.'
                ]
            }
        ])
        new_dataset = mapper.map(dataset)
        assert len(dataset) == len(new_dataset)     # same num dicts
        assert isinstance(new_dataset[0], dict)     # each element is dict
        assert 'input_ids' in new_dataset[0]        # dict has keys
        assert 'attention_mask' in new_dataset[0]
        assert len(new_dataset[0]['attention_mask']) == len(new_dataset[0]['input_ids'])    # mask same dimension as inputs
        assert self.tokenizer.decode(new_dataset[0]['input_ids'][0]) == '[CLS] this is a sentence. [SEP]'
        assert self.tokenizer.decode(new_dataset[0]['input_ids'][1]) == '[CLS] this is two sentences. here is the second one. [SEP]'

    def test_truncation_max_length(self):
        mapper = TokenizerMapper(
            input_field='text',
            tokenizer=self.tokenizer,
            truncation=True,
            max_length=10,
        )
        dataset = Dataset([
            {
                'text': [
                    'This is an instance that will be truncated because it is longer than ten word pieces.',
                    'This is the subsequent unit in this instance that will be separately truncated.'
                ]
            },
            {
                'text': [
                    'This is the next instance.'
                ]
            }
        ])
        new_dataset = mapper.map(dataset)
        assert len(dataset) == len(new_dataset)     # same num dicts
        assert self.tokenizer.decode(new_dataset[0]['input_ids'][0]) == '[CLS] this is an instance that will be truncated [SEP]'
        assert self.tokenizer.decode(new_dataset[0]['input_ids'][1]) == '[CLS] this is the subsequent unit in this instance [SEP]'
        assert self.tokenizer.decode(new_dataset[1]['input_ids'][0]) == '[CLS] this is the next instance. [SEP]'

    def test_overflow(self):
        mapper = TokenizerMapper(
            input_field='text',
            tokenizer=self.tokenizer,
            truncation=True,
            max_length=10,
            return_overflowing_tokens=True
        )
        dataset = Dataset([
            {
                'text': [
                    'This is an instance that will be truncated because it is longer than ten word pieces.',
                    'This is the subsequent unit in this instance that will be separately truncated.'
                ]
            },
            {
                'text': [
                    'This is the next instance.'
                ]
            }
        ])
        new_dataset = mapper.map(dataset)
        assert len(dataset) == len(new_dataset)     # same num dicts
        assert 'overflow_to_sample_mapping' in new_dataset[0]
        assert self.tokenizer.decode(new_dataset[0]['input_ids'][0]) == '[CLS] this is an instance that will be truncated [SEP]'
        assert self.tokenizer.decode(new_dataset[0]['input_ids'][1]) == '[CLS] because it is longer than ten word pieces [SEP]'
        assert self.tokenizer.decode(new_dataset[0]['input_ids'][2]) == '[CLS]. [SEP]'
        assert self.tokenizer.decode(new_dataset[0]['input_ids'][3]) == '[CLS] this is the subsequent unit in this instance [SEP]'
        assert self.tokenizer.decode(new_dataset[0]['input_ids'][4]) == '[CLS] that will be separately truncated. [SEP]'
        assert self.tokenizer.decode(new_dataset[1]['input_ids'][0]) == '[CLS] this is the next instance. [SEP]'
