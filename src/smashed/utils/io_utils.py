import os
import shutil
from contextlib import contextmanager
from logging import INFO, Logger, getLogger
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Optional,
    Union,
)
from urllib.parse import urlparse

from necessary import Necessary, necessary

with necessary("boto3", soft=True) as BOTO_AVAILABLE:
    if TYPE_CHECKING or BOTO_AVAILABLE:
        import boto3


__all__ = [
    'open_file_for_read',
    'open_file_for_write',
    'recursively_list_files',
    'remove_directory',
]


def get_logger() -> Logger:
    """Get the default logger for this module."""
    (logger := getLogger(__file__)).setLevel(INFO)
    return logger


@Necessary("boto3")
@contextmanager
def open_file_for_read(
    path: Union[str, Path],
    mode: str = "r",
    open_fn: Optional[Callable] = None,
    logger: Optional[Logger] = None,
    open_kwargs: Optional[Dict[str, Any]] = None,
) -> Generator[IO, None, None]:
    """Get a context manager to read in a file that is either on
    S3 or local.

    Args:
        path (Union[str, Path]): The path to the file to read. Can be an S3
            or local path.
        mode (str, optional): The mode to open the file in. Defaults  to "r".
            Only read modes are supported (e.g. 'rb', 'rt', 'r').
        open_fn (Callable, optional): The function to use to  open the file.
            Defaults to the built-in open function.
        logger (Logger, optional): The logger to use. Defaults to the built-in
            logger at INFO level.
        open_kwargs (Dict[str, Any], optional): Any additional keyword to pass
            to the open function. Defaults to None.
    """
    open_kwargs = open_kwargs or {}
    logger = logger or get_logger()
    open_fn = open_fn or open
    parse = urlparse(str(path))
    remove = False

    assert "r" in mode, "Only read mode is supported"

    if parse.scheme == "s3":
        client = boto3.client("s3")
        logger.info(f"Downloading {path} to a temporary file")
        with NamedTemporaryFile(delete=False) as f:
            path = f.name
            client.download_fileobj(parse.netloc, parse.path.lstrip("/"), f)
            remove = True
    elif parse.scheme == "file" or parse.scheme == "":
        pass
    else:
        raise ValueError(f"Unsupported scheme {parse.scheme}")

    try:
        with open_fn(file=path, mode=mode, **open_kwargs) as f:
            yield f
    finally:
        if remove:
            os.remove(path)


@Necessary("boto3")
@contextmanager
def open_file_for_write(
    path: Union[str, Path],
    mode: str = "w",
    skip_if_empty: bool = False,
    open_fn: Optional[Callable] = None,
    logger: Optional[Logger] = None,
    open_kwargs: Optional[Dict[str, Any]] = None,
) -> Generator[IO, None, None]:
    """Get a context manager to write to a file that is either on
    S3 or local.

    Args:
        path (Union[str, Path]): The path to the file to write. Can be local
            or an S3 path.
        mode (str, optional): The mode to open the file in. Defaults  to "w".
            Only read modes are supported (e.g. 'wb', 'w', ...).
        open_fn (Callable, optional): The function to use to  open the file.
            Defaults to the built-in open function.
        logger (Logger, optional): The logger to use. Defaults to the built-in
            logger at INFO level.
        open_kwargs (Dict[str, Any], optional): Any additional keyword to pass
            to the open function. Defaults to None.
    """

    parse = urlparse(str(path))
    local = None
    logger = logger or get_logger()
    open_fn = open_fn or open
    open_kwargs = open_kwargs or {}

    assert "w" in mode or "a" in mode, "Only write/append mode is supported"

    try:
        if parse.scheme == "file" or parse.scheme == "":
            # make enclosing directory if it doesn't exist
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            with open_fn(file=path, mode=mode, **open_kwargs) as f:
                yield f
        else:
            with NamedTemporaryFile(delete=False, mode=mode) as f:
                yield f
                local = f.name
    finally:
        if local is None:
            if skip_if_empty and os.stat(path).st_size == 0:
                logger.info(f"Skipping empty file {path}")
                os.remove(path)
        elif parse.scheme == "s3":
            dst = f'{parse.netloc}{parse.path.lstrip("/")}'
            if skip_if_empty and os.stat(local).st_size == 0:
                logger.info(f"Skipping upload to {dst} since {local} is empty")
            else:
                logger.info(f"Uploading {local} to {dst}")
                client = boto3.client("s3")
                client.upload_file(local, parse.netloc, parse.path.lstrip("/"))
            os.remove(local)
        else:
            raise ValueError(f"Unsupported scheme {parse.scheme}")


@Necessary("boto3")
def recursively_list_files(
    path: Union[str, Path], ignore_hidden_files: bool = True
) -> Iterable[str]:
    """Recursively list all files in the given directory on network prefix

    Args:
        path (Union[str, Path]): The path to list content at. Can be local
            or an S3 path.
        ignore_hidden_files (bool, optional): Whether to ignore hidden files
            (i.e. files that start with a dot) when listing. Defaults to True.
    """

    parse = urlparse(str(path))

    if parse.scheme == "s3":
        cl = boto3.client("s3")
        prefixes = [parse.path.lstrip("/")]

        while len(prefixes) > 0:
            prefix = prefixes.pop()
            paginator = cl.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=parse.netloc, Prefix=prefix)
            for page in pages:
                for obj in page["Contents"]:
                    if obj["Key"][-1] == "/":
                        prefixes.append(obj["Key"])
                    else:
                        yield f's3://{parse.netloc}/{obj["Key"]}'

    elif parse.scheme == "file" or parse.scheme == "":
        for root, _, files in os.walk(parse.path):
            for f in files:
                if ignore_hidden_files and f.startswith("."):
                    continue
                yield os.path.join(root, f)
    else:
        raise NotImplementedError(f"Unknown scheme: {parse.scheme}")


@Necessary("boto3")
def copy_directory(
    src: Union[str, Path],
    dst: Union[str, Path],
    ignore_hidden_files: bool = True,
    logger: Optional[Logger] = None,
):
    """Copy a directory from one location to another. Source or target
    locations can be local, remote, or a mix of both.

    Args:
        src (Union[str, Path]): The location to copy from. Can be local
            or a location on S3.
        dst (Union[str, Path]): The location to copy to. Can be local or S3.
        ignore_hidden_files (bool, optional): Whether to ignore hidden files
            on copy. Defaults to True.
        logger (Logger, optional): The logger to use. Defaults to the built-in
            logger at INFO level.
    """

    logger = logger or get_logger()

    src = Path(src)
    dst = Path(dst)

    cnt = 0

    for source_path in recursively_list_files(
        src, ignore_hidden_files=ignore_hidden_files
    ):
        destination = dst / Path(source_path).relative_to(
            src
        )

        logger.info(f"Copying {source_path} to {destination}; {cnt:,} so far")

        with open_file_for_read(source_path, mode="rb") as s:
            with open_file_for_write(destination, mode="wb") as d:
                d.write(s.read())

        cnt += 1


@Necessary("boto3")
def remove_directory(path: Union[str, Path]):
    """Completely remove a directory at the provided path."""

    parse = urlparse(str(path))

    if parse.scheme == "s3":
        client = boto3.client("s3")
        for fn in recursively_list_files(path, ignore_hidden_files=False):
            parsed = urlparse(str(fn))
            client.delete_object(
                Bucket=parsed.netloc, Key=parsed.path.lstrip("/")
            )
    elif parse.scheme == "file" or parse.scheme == "":
        shutil.rmtree(path, ignore_errors=True)
    else:
        raise NotImplementedError(f"Unknown scheme: {parse.scheme}")
