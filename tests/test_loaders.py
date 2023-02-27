import os
from pathlib import Path
from unittest import TestCase

from necessary import necessary

from smashed.mappers.loaders import (
    CsvLoaderMapper,
    HuggingFaceDatasetLoaderMapper,
)

FIXTURES_PATH = Path(__file__).parent / "fixtures"


class TestHuggingfaceLoader(TestCase):
    def test_huggingface_loader(self):
        necessary("datasets")  # make sure datasets is installed

        # need to sent this env var to make sure we don't we don't
        # raise an error when using tokenizers
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        mapper = HuggingFaceDatasetLoaderMapper()

        dataset = [
            {
                "path": "csv",
                "delimiter": "\t",
                "column_names": ["label", "question", "answer"],
                "data_files": {
                    "train": str(FIXTURES_PATH / "TrecQA" / "train.tsv.gz"),
                    "test": str(FIXTURES_PATH / "TrecQA" / "test.tsv.gz"),
                    "dev": str(FIXTURES_PATH / "TrecQA" / "dev.tsv.gz"),
                },
                "split": "train",
            }
        ]

        transformed_dataset = mapper.map(dataset)
        self.assertEqual(len(transformed_dataset.features), 3)
        self.assertEqual(
            sorted(transformed_dataset.features.keys()),
            ["answer", "label", "question"],
        )
        self.assertEqual(len(transformed_dataset), 53417)

    def test_csv_loader(self):
        necessary("smart_open")  # make sure smart_open is installed

        mapper = CsvLoaderMapper(
            paths_field="files",
            headers=["label", "question", "answer"],
            delimiter="\t",
        )

        dataset = mapper.map(
            [{"files": str(FIXTURES_PATH / "TrecQA" / "train.tsv.gz")}]
        )
        self.assertEqual(len(dataset), 53417)
        self.assertEqual(
            dataset[0]["question"],
            (
                "who is the author of the book , `` the iron lady : "
                "a biography of margaret thatcher '' ?"
            ),
        )
        self.assertEqual(
            dataset[0]["answer"],
            (
                "the iron lady ; a biography of margaret thatcher by hugo"
                " young -lrb- farrar , straus & giroux -rrb-"
            ),
        )
        self.assertEqual(dataset[0]["label"], "1")
