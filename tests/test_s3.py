import unittest
from logging import getLogger

import boto3
import moto

from smashed.utils.io_utils import (
    open_file_for_read,
    open_file_for_write,
    stream_file_for_read,
)


class TestIo(unittest.TestCase):
    mock_s3 = moto.mock_s3()
    BUCKET_NAME = "mytestbucket"
    FILE_KEY = "test.jsonl"
    CONTENT = "This is a test\nWith multiple lines\nBye!"
    REGION = "us-east-1"

    def setUp(self):
        self.mock_s3.start()
        self.conn = boto3.resource("s3", region_name=self.REGION)
        self.conn.create_bucket(Bucket=self.BUCKET_NAME)
        self.client = boto3.client("s3", region_name=self.REGION)
        getLogger("botocore").setLevel("INFO")

    def tearDown(self):
        self.mock_s3.stop()

    @property
    def PREFIX(self):
        return f"s3://{self.BUCKET_NAME}/{self.FILE_KEY}"

    def _write_file(self):
        self.client.put_object(
            Bucket=self.BUCKET_NAME, Key=self.FILE_KEY, Body=self.CONTENT
        )

    def _read_file(self):
        r = self.client.get_object(Bucket=self.BUCKET_NAME, Key=self.FILE_KEY)
        return r["Body"].read().decode("utf-8")

    def test_read_from_s3(self):
        self._write_file()
        with open_file_for_read(self.PREFIX) as f:
            self.assertEqual(f.read(), self.CONTENT)

    def test_write_to_s3(self):
        with open_file_for_write(self.PREFIX) as f:
            f.write(self.CONTENT)

        content = self._read_file()
        self.assertEqual(content, self.CONTENT)

    def test_stream_from_s3(self):
        self._write_file()
        with stream_file_for_read(self.PREFIX) as f:
            self.assertEqual(f.read(), self.CONTENT)

    def test_stream_lines_from_s3(self):
        self._write_file()
        with stream_file_for_read(self.PREFIX) as f:
            for la, lb in zip(f, self.CONTENT.split("\n")):
                self.assertEqual(la.strip(), lb)
