import re
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union
from urllib.parse import urlparse

from necessary import necessary

with necessary("boto3", soft=True) as BOTO_AVAILABLE:
    if TYPE_CHECKING or BOTO_AVAILABLE:
        from botocore.client import BaseClient


PathType = Union[str, Path, "MultiPath"]
ClientType = Union["BaseClient", None]

LOGGER = getLogger(__file__)


@dataclass
class MultiPath:
    """A path object that can handle both local and remote paths."""

    prot: str
    root: str
    path: str

    def __post_init__(self):
        SUPPORTED_PROTOCOLS = {"s3", "file"}
        if self.prot and self.prot not in SUPPORTED_PROTOCOLS:
            raise ValueError(
                f"Unsupported protocol: {self.prot}; "
                f"supported protocols are {SUPPORTED_PROTOCOLS}"
            )

    @classmethod
    def parse(cls, path: PathType) -> "MultiPath":
        """Parse a path into a PathParser object.

        Args:
            path (str): The path to parse.
        """
        if isinstance(path, cls):
            return path
        elif isinstance(path, Path):
            path = str(path)
        elif not isinstance(path, str):
            raise ValueError(f"Cannot parse path of type {type(path)}")

        p = urlparse(str(path))
        return cls(prot=p.scheme, root=p.netloc, path=p.path)

    @property
    def is_s3(self) -> bool:
        """Is true if the path is an S3 path."""
        return self.prot == "s3"

    @property
    def is_local(self) -> bool:
        """Is true if the path is a local path."""
        return self.prot == "file" or self.prot == ""

    def _remove_extra_slashes(self, path: str) -> str:
        return re.sub(r"//+", "/", path)

    def __str__(self) -> str:
        if self.prot:
            loc = self._remove_extra_slashes(f"{self.root}/{self.path}")
            return f"{self.prot}://{loc}"
        elif self.root:
            return self._remove_extra_slashes(f"/{self.root}/{self.path}")
        else:
            return self._remove_extra_slashes(self.path)

    @property
    def bucket(self) -> str:
        """If the path is an S3 path, return the bucket name.
        Otherwise, raise a ValueError."""
        if not self.is_s3:
            raise ValueError(f"Not an S3 path: {self}")
        return self.root

    @property
    def key(self) -> str:
        """If the path is an S3 path, return the prefix.
        Otherwise, raise a ValueError."""
        if not self.is_s3:
            raise ValueError(f"Not an S3 path: {self}")
        return self.path.lstrip("/")

    @property
    def as_path(self) -> Path:
        """Return the path as a pathlib.Path object."""
        if not self.is_local:
            raise ValueError(f"Not a local path: {self}")
        return Path(self.as_str)

    def __hash__(self) -> int:
        return hash(self.as_str)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, (MultiPath, str, Path)):
            return False

        other = MultiPath.parse(other)
        return self.as_str == other.as_str

    @property
    def as_str(self) -> str:
        """Return the path as a string."""
        return str(self)

    def __truediv__(self, other: PathType) -> "MultiPath":
        """Join two paths together using the / operator."""
        other = MultiPath.parse(other)

        if isinstance(other, MultiPath) and other.prot:
            raise ValueError(f"Cannot combine fully formed path {other}")

        return MultiPath(
            prot=self.prot,
            root=self.root,
            path=f"{self.path.rstrip('/')}/{str(other).lstrip('/')}",
        )

    def __len__(self) -> int:
        return len(self.as_str)

    def __sub__(self, other: PathType) -> "MultiPath":
        _o_str = MultiPath.parse(other).as_str
        _s_str = self.as_str
        loc = _s_str.find(_o_str)
        return MultiPath.parse(_s_str[:loc] + _s_str[loc + len(_o_str) :])

    @classmethod
    def join(cls, *others: PathType) -> "MultiPath":
        """Join multiple paths together; each path can be a string,
        pathlib.Path, or MultiPath object."""
        if not others:
            raise ValueError("No paths provided")

        first, *rest = others
        first = cls.parse(first)
        for part in rest:
            # explicitly call __div__ to avoid mypy errors
            first = first / part
        return first
