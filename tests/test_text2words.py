from unittest import TestCase

from smashed.mappers.text import TextToWordsMapper, WordsToTextMapper


class TestText2Words(TestCase):
    def test_trail(self):
        mapper = TextToWordsMapper(
            fields="text", splitter="trail"
        ) >> WordsToTextMapper(fields="text", joiner="")
        text = "Hello world! What a beautiful day...\nOR NOT?"
        dataset = [{"text": text}]
        mapped_dataset = mapper.map(dataset)
        self.assertEqual(mapped_dataset[0]["text"], text)
