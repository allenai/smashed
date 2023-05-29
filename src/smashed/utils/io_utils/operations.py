import io
import shutil
from contextlib import ExitStack, contextmanager
from logging import Logger, getLogger
from os import remove as remove_local_file
from os import stat as stat_local_file
from os import walk as local_walk
from pathlib import Path
from tempfile import NamedTemporaryFile, gettempdir
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
    cast,
)

from necessary import necessary

from .io_wrappers import ReadBytesIO, ReadTextIO
from .multipath import MultiPath

with necessary("boto3", soft=True) as BOTO_AVAILABLE:
    if TYPE_CHECKING or BOTO_AVAILABLE:
        import boto3
        from botocore.client import BaseClient


PathType = Union[str, Path, "MultiPath"]
ClientType = Union["BaseClient", None]

LOGGER = getLogger(__file__)


def get_client_if_needed(path: PathType, **boto3_kwargs: Any) -> ClientType:
    """Return the appropriate client given the protocol of the path."""

    path = MultiPath.parse(path)

    if path.is_s3:
        # necessary here will raise an error if boto3 is not installed.
        with necessary(
            "boto3",
            message=(
                "{module_name} is required for S3 support;"
                "run 'pip install smashed[remote]' or 'pip install boto3'."
            ),
        ):
            return boto3.client("s3", **boto3_kwargs)  # pyright: ignore

    return None  # pyright: ignore


def get_temp_dir(path: Optional[PathType]) -> Path:
    """Check if the directory `path` can be used as a temporary directory."""

    if path is None:
        # return the default temporary directory
        return Path(gettempdir())

    path = MultiPath.parse(path)
    if not path.is_local:
        raise ValueError(f"Temporary directory must be local: {path}")

    path = Path(str(path))
    if path.exists() and not path.is_dir():
        raise ValueError(f"Temporary directory must be a directory: {path}")

    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def stream_file_for_read(
    path: PathType,
    mode: str = "r",
    open_fn: Optional[Callable] = None,
    logger: Optional[Logger] = None,
    open_kwargs: Optional[Dict[str, Any]] = None,
    client: Optional[ClientType] = None,
) -> Generator[IO, None, None]:
    """Just like open_file_for_read, but returns a file-like object that
    streams content from remote files instead of saving it locally first.

    Args:
        path (Union[str, Path, MultiPath]): The path to the file to read.
        mode (str, optional): The mode to open the file in. Defaults to "r".
        open_fn (Callable, optional): The function to use to  open the file.
            Defaults to the built-in open function.
        logger (Logger, optional): The logger to use. Defaults to the built-in
            logger at INFO level.
        open_kwargs (Dict[str, Any], optional): Any additional keyword to pass
            to the open function. Defaults to None.
        client (ClientType, optional): The client to use to download the file.
            If not provided, one will be created using the default boto3
            if necessary. Defaults to None.
    """

    open_kwargs = open_kwargs or {}
    logger = logger or LOGGER
    open_fn = open_fn or open

    assert "r" in mode, "Only read mode is supported"

    path = MultiPath.parse(path)

    if path.is_s3:
        client = client or get_client_if_needed(path)
        assert client is not None, "Could not get S3 client"

        obj = client.get_object(Bucket=path.bucket, Key=path.key.lstrip("/"))

        stream: io.IOBase
        if "b" in mode:
            stream = ReadBytesIO(obj["Body"])
        else:
            stream = ReadTextIO(obj["Body"])

        yield cast(IO, stream)
    elif path.is_local:
        with open_fn(file=path.as_str, mode=mode, **open_kwargs) as f:
            yield f
    else:
        raise ValueError(f"Unsupported protocol: {path.prot}")


@contextmanager
def open_file_for_read(
    path: PathType,
    mode: str = "r",
    open_fn: Optional[Callable] = None,
    logger: Optional[Logger] = None,
    open_kwargs: Optional[Dict[str, Any]] = None,
    client: Optional[ClientType] = None,
    temp_dir: Optional[PathType] = None,
) -> Generator[IO, None, None]:
    """Get a context manager to read in a file that is either in a local
    or remote location. If the path is a remote path, the file will be
    downloaded to a temporary location and then deleted after the context
    manager exits.

    Args:
        path (Union[str, Path, MultiPath]): The path to the file to read.
        mode (str, optional): The mode to open the file in. Defaults  to "r".
            Only read modes are supported (e.g. 'rb', 'rt', 'r').
        open_fn (Callable, optional): The function to use to  open the file.
            Defaults to the built-in open function.
        logger (Logger, optional): The logger to use. Defaults to the built-in
            logger at INFO level.
        open_kwargs (Dict[str, Any], optional): Any additional keyword to pass
            to the open function. Defaults to None.
        client (ClientType, optional): The client to use to download the file.
            If not provided, one will be created using the default boto3
            if necessary. Defaults to None.
        temp_dir (Union[str, Path, MultiPath], optional): The directory to
            download the file to. Defaults to None, which will use the
            system default.
    """
    open_kwargs = open_kwargs or {}
    logger = logger or LOGGER
    open_fn = open_fn or open
    remove = False

    assert "r" in mode, "Only read mode is supported"

    path = MultiPath.parse(path)

    if path.is_s3:
        client = client or get_client_if_needed(path)
        assert client is not None, "Could not get S3 client"

        logger.info(f"Downloading {path} to a temporary file")
        with NamedTemporaryFile(delete=False, dir=get_temp_dir(temp_dir)) as f:
            client.download_fileobj(path.bucket, path.key.lstrip("/"), f)
            path = MultiPath.parse(f.name)
            remove = True
    try:
        with open_fn(file=str(path), mode=mode, **open_kwargs) as f:
            yield f
    finally:
        if remove:
            remove_local_file(str(path))


def is_dir(
    path: PathType,
    client: Optional[ClientType] = None,
    raise_if_not_exists: bool = False,
) -> bool:
    """Check if a path is a directory."""

    path = MultiPath.parse(path)
    client = client or get_client_if_needed(path)

    if path.is_local:
        if not (e := path.as_path.exists()) and raise_if_not_exists:
            raise FileNotFoundError(f"Path does not exist: {path}")
        elif not e:
            return False
        return path.as_path.is_dir()
    elif path.is_s3:
        assert client is not None, "Could not get S3 client"
        resp = client.list_objects_v2(
            Bucket=path.bucket, Prefix=path.key.lstrip("/"), Delimiter="/"
        )
        if "CommonPrefixes" in resp:
            return True
        elif "Contents" in resp:
            return False
        elif raise_if_not_exists:
            raise FileNotFoundError(f"Path does not exist: {path}")
        return False
    else:
        raise FileNotFoundError(f"Unsupported protocol: {path.prot}")


def is_file(
    path: PathType,
    client: Optional[ClientType] = None,
    raise_if_not_exists: bool = False,
) -> bool:
    """Check if a path is a file."""

    try:
        return not is_dir(path=path, client=client, raise_if_not_exists=True)
    except FileNotFoundError as e:
        if raise_if_not_exists:
            raise FileNotFoundError(f"Path does not exist: {path}") from e
        return False


def exists(
    path: PathType,
    client: Optional[ClientType] = None,
) -> bool:
    """Check if a path exists"""

    try:
        is_dir(path=path, client=client, raise_if_not_exists=True)
        return True
    except FileNotFoundError:
        return False


@contextmanager
def open_file_for_write(
    path: PathType,
    mode: str = "w",
    skip_if_empty: bool = False,
    open_fn: Optional[Callable] = None,
    logger: Optional[Logger] = None,
    open_kwargs: Optional[Dict[str, Any]] = None,
    client: Optional[ClientType] = None,
    temp_dir: Optional[PathType] = None,
) -> Generator[IO, None, None]:
    """Get a context manager to write to a file. If the file is from a
    remote location (e.g. S3), the file will be written to a temporary
    file and then uploaded to the remote location; after the context
    manager exits, the temporary file will be deleted.

    Args:
        path (Union[str, Path, MultiPath]): The path to the file to write.
        mode (str, optional): The mode to open the file in. Defaults  to "w".
            Only read modes are supported (e.g. 'wb', 'w', ...).
        skip_if_empty (bool, optional): If True, the file will not be
            written if the content is empty. Defaults to False.
        open_fn (Callable, optional): The function to use to  open the file.
            Defaults to the built-in open function.
        logger (Logger, optional): The logger to use. Defaults to the built-in
            logger at INFO level.
        open_kwargs (Dict[str, Any], optional): Any additional keyword to pass
            to the open function. Defaults to None.
        client (boto3.client, optional): The boto3 client to use. If not
            provided, one will be created if necessary.
        temp_dir (Union[str, Path, MultiPath], optional): The directory to
            use for temporary files. Defaults to the system temp directory.
    """

    path = str(path)
    local = None
    logger = logger or LOGGER
    open_fn = open_fn or open
    open_kwargs = open_kwargs or {}

    path = MultiPath.parse(path)

    assert "w" in mode or "a" in mode, "Only write/append mode is supported"

    try:
        if path.is_local:
            # make enclosing directory if it doesn't exist
            path.as_path.parent.mkdir(parents=True, exist_ok=True)

            with open_fn(file=str(path), mode=mode, **open_kwargs) as f:
                yield f
        else:
            with NamedTemporaryFile(
                delete=False, mode=mode, dir=get_temp_dir(temp_dir)
            ) as f:
                yield f
                local = MultiPath.parse(f.name)
    finally:
        if local is None:
            if skip_if_empty and stat_local_file(path.as_str).st_size == 0:
                logger.info(f"Skipping empty file {path}")
                remove_local_file(path.as_path)
        elif path.is_s3:
            # dst = f'{path.bucket}{parse.path.lstrip("/")}'
            if skip_if_empty and stat_local_file(local.as_str).st_size == 0:
                logger.info(f"Skipping upload to {path}: {local} is empty")
            else:
                logger.info(f"Uploading {local} to {path}")
                client = client or get_client_if_needed(path)
                assert client is not None, "Could not get S3 client"
                client.upload_file(
                    local.as_str, path.bucket, path.key.lstrip("/")
                )
            remove_local_file(local.as_path)


def recursively_list_files(
    path: PathType,
    ignore_hidden: bool = True,
    include_dirs: bool = False,
    include_files: bool = True,
    client: Optional[ClientType] = None,
    local_follow_links: bool = False,
) -> Iterable[str]:
    """Recursively list all files in the given directory for a given
    path, local or remote.

    Args:
        path (Union[str, Path, MultiPath]): The path to list content at.
        ignore_hidden (bool, optional): Whether to ignore hidden files and
            directories when listing. Defaults to True.
        include_dirs (bool, optional): Whether to include directories in the
            listing. Defaults to False.
        include_files (bool, optional): Whether to include files in the
            listing. Defaults to True.
        client (boto3.client, optional): The boto3 client to use. If not
            provided, one will be created if necessary.
        local_follow_links (bool, optional): Whether to follow symlinks when
            listing local files. Defaults to False.
    """

    path = MultiPath.parse(path)

    if path.is_s3:
        client = client or get_client_if_needed(path)
        assert client is not None, "Could not get S3 client"

        prefixes = [path.key.lstrip("/")]

        while len(prefixes) > 0:
            prefix = prefixes.pop()
            paginator = client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=path.bucket, Prefix=prefix)
            for page in pages:
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    path = MultiPath(prot="s3", root=path.root, path=key)
                    if key[-1] == "/" and key != prefix:
                        # last char is a slash, so it's a directory
                        # we don't want to re-include the prefix though, so we
                        # check that it's not the same
                        prefixes.append(key)
                        if include_dirs:
                            yield str(path)
                    else:
                        if include_files:
                            yield str(path)

    if path.is_local:
        if not path.as_path.is_dir():
            # yield the path itself if it's not a directory; this matches
            # the behavior that we get for S3.
            yield path.as_str

        for _root, dirnames, filenames in local_walk(
            top=path.as_str, followlinks=local_follow_links
        ):
            root = Path(_root)
            to_list = [
                *(dirnames if include_dirs else []),
                *(filenames if include_files else []),
            ]
            for f in to_list:
                if ignore_hidden and f.startswith("."):
                    continue
                yield str(MultiPath.parse(root / f))


def copy_directory(
    src: PathType,
    dst: PathType,
    ignore_hidden_files: bool = False,
    skip_if_empty: bool = False,
    logger: Optional[Logger] = None,
    client: Optional[ClientType] = None,
):
    """Copy a directory from one location to another. Source or target
    locations can be local, remote, or a mix of both.

    Args:
        src (Union[str, Path, MultiPath]): The location to copy from.
        dst (Union[str, Path, MultiPath]): The location to copy to.
        ignore_hidden_files (bool, optional): Whether to ignore hidden files
            on copy. Defaults to True.
        logger (Logger, optional): The logger to use. Defaults to the built-in
            logger at INFO level.
    """

    logger = logger or LOGGER

    # we convert to string because the Path library does not handle
    # well network locations.
    src = MultiPath.parse(src)
    dst = MultiPath.parse(dst)
    cnt = 0

    client = client or get_client_if_needed(src) or get_client_if_needed(dst)

    for sp in recursively_list_files(
        path=src, ignore_hidden=ignore_hidden_files
    ):
        # parse the source path
        source_path = MultiPath.parse(sp)

        # we strip the segment of source_path that is the
        # common prefix in src, then join the remaining bit
        destination = dst / (source_path - src)

        logger.info(f"Copying {source_path} to {destination}; {cnt:,} so far")

        with ExitStack() as stack:
            s = stack.enter_context(
                open_file_for_read(source_path, mode="rb", client=client)
            )
            d = stack.enter_context(
                open_file_for_write(
                    destination,
                    mode="wb",
                    skip_if_empty=skip_if_empty,
                    client=client,
                )
            )
            d.write(s.read())

        cnt += 1


def remove_file(path: PathType, client: Optional[ClientType] = None):
    """Remove a file at the provided path."""

    path = MultiPath.parse(path)

    if path.is_s3:
        client = client or get_client_if_needed(path)
        assert client is not None, "Could not get S3 client"
        client.delete_object(Bucket=path.bucket, Key=path.key.lstrip("/"))

    if path.is_local:
        remove_local_file(path.as_path)


def remove_directory(path: PathType, client: Optional[ClientType] = None):
    """Completely remove a directory at the provided path."""

    path = MultiPath.parse(path)

    if path.is_s3:
        client = client or get_client_if_needed(path)
        assert client is not None, "Could not get S3 client"

        for fn in recursively_list_files(
            path=path, ignore_hidden=False, client=client
        ):
            remove_file(fn, client=client)

    if path.is_local:
        shutil.rmtree(path.as_str, ignore_errors=True)
