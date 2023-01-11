import shutil
from contextlib import AbstractContextManager, ExitStack, contextmanager
from functools import partial
from logging import INFO, Logger, getLogger
from os import remove as remove_local_file
from os import stat as stat_local_file
from os import walk as local_walk
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Optional,
    TypeVar,
    Union,
)
from urllib.parse import ParseResult, urlparse

from necessary import necessary
from typing_extensions import Concatenate, ParamSpec

with necessary("boto3", soft=True) as BOTO_AVAILABLE:
    if TYPE_CHECKING or BOTO_AVAILABLE:
        import boto3
        from botocore.client import BaseClient


__all__ = [
    "copy_directory",
    "open_file_for_read",
    "open_file_for_write",
    "recursively_list_files",
    "remove_directory",
    "remove_file",
    "upload_on_success",
]

PathType = Union[str, Path, ParseResult]
ClientType = Union["BaseClient", None]


def uri_stringify(uri: PathType) -> str:
    """Convert a URI to a string."""
    if isinstance(uri, str):
        return uri

    if isinstance(uri, Path):
        return str(uri)

    if isinstance(uri, ParseResult):
        return uri.geturl()


def join_uri(*uris: PathType) -> str:
    """Join a URI."""
    first, *rest, last = map(uri_stringify, uris)
    rest = [part.strip("/") for part in rest]
    return "/".join([first.rstrip("/"), *rest, last.lstrip("/")])


def get_logger() -> Logger:
    """Get the default logger for this module."""
    (logger := getLogger(__file__)).setLevel(INFO)
    return logger


def get_client_if_needed(path: PathType) -> ClientType:
    parse = (
        urlparse(uri_stringify(path))
        if not isinstance(path, ParseResult)
        else path
    )

    if parse.scheme == "s3":
        # necessary here will raise an error if boto3 is not installed.
        with necessary(
            "boto3",
            message=(
                "{module_name} is required for S3 support;"
                "run 'pip install smashed[remote]' or 'pip install boto3'."
            ),
        ):
            return boto3.client("s3")  # pyright: ignore
    elif parse.scheme == "file" or parse.scheme == "":
        return None  # pyright: ignore
    else:
        raise ValueError(f"Unsupported scheme {parse.scheme}")


@contextmanager
def open_file_for_read(
    path: Union[str, Path],
    mode: str = "r",
    open_fn: Optional[Callable] = None,
    logger: Optional[Logger] = None,
    open_kwargs: Optional[Dict[str, Any]] = None,
    client: Optional[ClientType] = None,
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
        client = client or get_client_if_needed(path)
        assert client is not None, "Could not get S3 client"

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
            remove_local_file(path)


@contextmanager
def open_file_for_write(
    path: Union[str, Path],
    mode: str = "w",
    skip_if_empty: bool = False,
    open_fn: Optional[Callable] = None,
    logger: Optional[Logger] = None,
    open_kwargs: Optional[Dict[str, Any]] = None,
    client: Optional[ClientType] = None,
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

    path = str(path)
    parse = urlparse(path)
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
            if skip_if_empty and stat_local_file(path).st_size == 0:
                logger.info(f"Skipping empty file {path}")
                remove_local_file(path)
        elif parse.scheme == "s3":
            dst = f'{parse.netloc}{parse.path.lstrip("/")}'
            if skip_if_empty and stat_local_file(local).st_size == 0:
                logger.info(f"Skipping upload to {dst} since {local} is empty")
            else:
                logger.info(f"Uploading {local} to {dst}")
                client = client or get_client_if_needed(path)
                assert client is not None, "Could not get S3 client"
                client.upload_file(local, parse.netloc, parse.path.lstrip("/"))
            remove_local_file(local)
        else:
            raise ValueError(f"Unsupported scheme {parse.scheme}")


def recursively_list_files(
    path: Union[str, Path],
    ignore_hidden_files: bool = True,
    client: Optional[ClientType] = None,
) -> Iterable[str]:
    """Recursively list all files in the given directory on network prefix

    Args:
        path (Union[str, Path]): The path to list content at. Can be local
            or an S3 path.
        ignore_hidden_files (bool, optional): Whether to ignore hidden files
            (i.e. files that start with a dot) when listing. Defaults to True.
    """

    path = str(path)
    parse = urlparse(path)

    if parse.scheme == "s3":
        client = client or get_client_if_needed(path)
        assert client is not None, "Could not get S3 client"

        prefixes = [parse.path.lstrip("/")]

        while len(prefixes) > 0:
            prefix = prefixes.pop()
            paginator = client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=parse.netloc, Prefix=prefix)
            for page in pages:
                for obj in page["Contents"]:
                    if obj["Key"][-1] == "/":
                        prefixes.append(obj["Key"])
                    else:
                        yield f's3://{parse.netloc}/{obj["Key"]}'

    elif parse.scheme == "file" or parse.scheme == "":
        for root, _, files in local_walk(parse.path):
            for f in files:
                if ignore_hidden_files and f.startswith("."):
                    continue
                yield join_uri(root, f)
    else:
        raise NotImplementedError(f"Unknown scheme: {parse.scheme}")


def copy_directory(
    src: Union[str, Path],
    dst: Union[str, Path],
    ignore_hidden_files: bool = False,
    skip_if_empty: bool = False,
    logger: Optional[Logger] = None,
    client: Optional[ClientType] = None,
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

    # we convert to string because the Path library does not handle
    # well network locations.
    src = str(src)
    dst = str(dst)
    cnt = 0

    client = client or get_client_if_needed(src) or get_client_if_needed(dst)

    for source_path in recursively_list_files(
        str(src), ignore_hidden_files=ignore_hidden_files
    ):
        # we strip the segment of source_path that is the common prefix in src,
        # then join the remaining bit
        destination = join_uri(dst, source_path[len(src) :])

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


def remove_file(path: Union[str, Path], client: Optional[ClientType] = None):
    """Remove a file at the provided path."""

    path = str(path)
    parse = urlparse(path)

    if parse.scheme == "s3":
        client = client or get_client_if_needed(path)
        assert client is not None, "Could not get S3 client"
        client.delete_object(Bucket=parse.netloc, Key=parse.path.lstrip("/"))
    elif parse.scheme == "file" or parse.scheme == "":
        remove_local_file(path)
    else:
        raise NotImplementedError(f"Unknown scheme: {parse.scheme}")


def remove_directory(
    path: Union[str, Path], client: Optional[ClientType] = None
):
    """Completely remove a directory at the provided path."""

    parse = urlparse(str(path))

    if parse.scheme == "s3":
        client = client or get_client_if_needed(path)
        assert client is not None, "Could not get S3 client"

        for fn in recursively_list_files(
            path=path, ignore_hidden_files=False, client=client
        ):
            remove_file(fn, client=client)
    elif parse.scheme == "file" or parse.scheme == "":
        shutil.rmtree(path, ignore_errors=True)
    else:
        raise NotImplementedError(f"Unknown scheme: {parse.scheme}")


T = TypeVar("T")
P = ParamSpec("P")


class upload_on_success(AbstractContextManager):
    """Context manager to upload a directory of results to a remote
    location if the execution in the context manager is successful.

    You can use this class in two ways:

    1. As a context manager

        ```python

        with upload_on_success('s3://my-bucket/my-results') as path:
            # run training, save temporary results in `path`
            ...
        ```

    2. As a function decorator

        ```python
        @upload_on_success('s3://my-bucket/my-results')
        def my_function(path: str, ...)
            # run training, save temporary results in `path`
        ```

    You can specify a local destination by passing `local_path` to
    `upload_on_success`. Otherwise, a temporary directory is created for  you.
    """

    def __init__(
        self,
        remote_path: PathType,
        local_path: Optional[PathType] = None,
        keep_local: bool = False,
    ) -> None:
        """Constructor for upload_on_success context manager

        Args:
            remote_path (str or urllib.parse.ParseResult): The remote location
                to upload to (e.g., an S3 prefix for a bucket you have
                access to).
            local_path (str or Path): The local path where to temporarily
                store files before upload. If not provided, a temporary
                directory is created for you and returned by the context
                manager. It will be deleted at the end of the context
                (unless keep_local is set to True). Defaults to None
            keep_local (bool, optional): Whether to keep the local results
                as well as uploading to the remote path. Only available
                if `local_path` is provided.
        """

        self._ctx = ExitStack()
        self.remote_path = remote_path
        self.local_path = (
            uri_stringify(local_path)
            if local_path is not None
            else self._ctx.enter_context(TemporaryDirectory())
        )
        if local_path is None and keep_local:
            raise ValueError(
                "Cannot keep local destination if `local_path` is None"
            )
        self.keep_local = keep_local

        super().__init__()

    def _decorated(
        self,
        func: Callable[Concatenate[str, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        with type(self)(
            local_path=self.local_path,
            remote_path=self.remote_path,
            keep_local=self.keep_local,
        ) as path:
            output = func(path, *args, **kwargs)
        return output

    def __call__(
        self, func: Callable[Concatenate[str, P], T]
    ) -> Callable[P, T]:
        return partial(self._decorated, func=func)  # type: ignore

    def __enter__(self):
        return self.local_path

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            # all went well, so we copy the local directory to the remote
            copy_directory(
                src=self.local_path, dst=self.remote_path  # pyright: ignore
            )

        if not self.keep_local:
            remove_directory(self.local_path)

        self._ctx.close()
