import unittest

from smashed.mappers.tokenize import ValidUnicodeMapper
from smashed.interfaces.simple import Dataset, ValidUnicodeMapper

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
