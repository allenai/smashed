import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from smashed.utils.io_utils import (
    MultiPath,
    copy_directory,
    recursively_list_files,
    remove_directory,
    remove_file,
)


class TestMultiPath(unittest.TestCase):
    def test_parse(self):
        s3_path = "s3://bucket/path/to/file"
        parse = MultiPath.parse(s3_path)
        self.assertEqual(parse.prot, "s3")
        self.assertEqual(parse.bucket, "bucket")
        self.assertEqual(parse.key, "path/to/file")
        self.assertEqual(str(parse), s3_path)

        local_path = "/path/to/file"
        parse = MultiPath.parse(local_path)
        self.assertEqual(parse.prot, "")
        self.assertEqual(str(parse), local_path)

        local_path = "path/to/file"
        parse = MultiPath.parse(local_path)
        self.assertEqual(parse.prot, "")
        self.assertEqual(str(parse), local_path)

        local_path = "file://path/to/file"
        parse = MultiPath.parse(local_path)
        self.assertEqual(parse.prot, "file")
        self.assertEqual(str(parse), local_path)

        gs_path = "gs://bucket/path/to/file"
        with self.assertRaises(ValueError):
            MultiPath.parse(gs_path)

    def test_join(self):
        self.assertEqual(
            MultiPath.parse("s3://bucket/path/to") / "new_file",
            MultiPath.parse("s3://bucket/path/to/new_file"),
        )

        self.assertEqual(
            MultiPath.parse("s3://bucket/path/to/") / "/new_file",
            MultiPath.parse("s3://bucket/path/to/new_file"),
        )

        self.assertEqual(
            MultiPath.join("foo", Path("bar"), MultiPath.parse("/baz")),
            MultiPath.parse("foo/bar/baz"),
        )

        with self.assertRaises(ValueError):
            _ = MultiPath.parse("s3://bucket/path/to") / "s3://bucket/path/to"

    def test_types(self):
        s3_path = MultiPath.parse("s3://bucket/path/to/file")
        self.assertTrue(s3_path.is_s3)
        self.assertFalse(s3_path.is_local)

        with self.assertRaises(ValueError):
            s3_path.as_path

        local_path = MultiPath.parse("/path/to/file")
        self.assertFalse(local_path.is_s3)
        self.assertTrue(local_path.is_local)

        with self.assertRaises(ValueError):
            local_path.bucket
            local_path.key

    def test_subtraction(self):
        path_a = MultiPath.parse("s3://bucket/path/to/file")
        path_b = MultiPath.parse("s3://bucket/")
        self.assertEqual((path_a - path_b).as_str, "path/to/file")
        self.assertEqual((path_b - path_a).as_str, "s3://bucket/")

    def test_local_operations(self):
        with TemporaryDirectory() as tmpdir:
            root_path = MultiPath.parse(tmpdir)

            # make a directory
            (root_path / "d1").as_path.mkdir()

            # make some files
            for file_name in ["f1", "f2"]:
                (root_path / "d1" / file_name).as_path.touch()

            # make some nested directories and files
            (root_path / "d1" / "d11").as_path.mkdir()
            (root_path / "d1" / "d11" / "f11").as_path.touch()

            # test listing functionality
            all_files = {f"{tmpdir}/d1/{f}" for f in ("f1", "f2", "d11/f11")}

            for fn in recursively_list_files(root_path / "d1"):
                self.assertIn(fn, all_files)

            # test copy
            copy_directory(root_path / "d1", root_path / "d2")

            all_files = {f"{tmpdir}/d2/{f}" for f in ("f1", "f2", "d11/f11")}
            for fn in recursively_list_files(root_path / "d2"):
                self.assertIn(fn, all_files)

            # test copy in a non-empty directory
            (root_path / "d3" / "d11").as_path.mkdir(parents=True)
            (root_path / "d3" / "d11" / "f11").as_path.touch()

            all_files = {f"{tmpdir}/d3/{f}" for f in ("f1", "f2", "d11/f11")}
            copy_directory(root_path / "d1", root_path / "d3")
            for fn in recursively_list_files(root_path / "d3"):
                self.assertIn(fn, all_files)

            # test remove
            remove_directory(root_path / "d1")
            self.assertFalse((root_path / "d1").as_path.exists())

            # test remove file
            remove_file(root_path / "d3" / "f1")
            self.assertFalse((root_path / "d3" / "f1").as_path.exists())

            with self.assertRaises(FileNotFoundError):
                remove_file(root_path / "d3" / "f1")
