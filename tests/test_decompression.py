import json
import unittest
from pathlib import Path

from smashed.utils.io_utils import decompress_stream

FIXTURES_PATH = Path(__file__).parent / "fixtures"


class TestDecompression(unittest.TestCase):
    def setUp(self) -> None:
        self.arxiv_path = FIXTURES_PATH / "compressed_jsonl" / "arxiv.gz"
        self.c4_train_path = FIXTURES_PATH / "compressed_jsonl" / "c4-train.gz"

    def test_bytes_compression(self):
        cnt = 0
        with open(self.arxiv_path, "rb") as f:
            with decompress_stream(f, "rb", gzip=True) as g:
                for ln in g:
                    json.loads(ln)
                    cnt += 1
        self.assertEqual(cnt, 9)

        cnt = 0
        with open(self.c4_train_path, "rb") as f:
            with decompress_stream(f, "rb", gzip=True) as g:
                for ln in g:
                    json.loads(ln)
                    cnt += 1
        self.assertEqual(cnt, 185)

    def test_text_compression(self):
        cnt = 0
        with open(self.arxiv_path, "rb") as f:
            with decompress_stream(f, gzip=True) as g:
                for ln in g:
                    json.loads(ln)
                    cnt += 1
        self.assertEqual(cnt, 9)

        cnt = 0
        with open(self.c4_train_path, "rb") as f:
            with decompress_stream(f, "rt", gzip=True) as g:
                for ln in g:
                    json.loads(ln)
                    cnt += 1
        self.assertEqual(cnt, 185)
