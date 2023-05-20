from contextlib import AbstractContextManager, ExitStack
from functools import partial
from tempfile import TemporaryDirectory
from typing import Callable, Optional, TypeVar

from typing_extensions import Concatenate, ParamSpec

from .multipath import MultiPath
from .operations import PathType, copy_directory, remove_directory

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
        self.local_path = MultiPath.parse(
            local_path or self._ctx.enter_context(TemporaryDirectory())
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
            output = func(path.as_str, *args, **kwargs)
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
            copy_directory(src=self.local_path, dst=self.remote_path)

        if not self.keep_local:
            remove_directory(self.local_path)

        self._ctx.close()
