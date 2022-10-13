import importlib.metadata
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Type, TypeVar, Union

import platformdirs

if TYPE_CHECKING:
    from .base import BatchedBaseMapper, SingleBaseMapper


M = TypeVar("M", "SingleBaseMapper", "BatchedBaseMapper")


def get_version() -> str:
    """Get the version of the package."""

    # This is a workaround for the fact that if the package is installed
    # in editable mode, the version is not reliability available.
    # Therefore, we check for the existence of a file called EDITABLE,
    # which is not included in the package at distribution time.
    path = Path(__file__).parent / "EDITABLE"
    if path.exists():
        return "dev"

    try:
        # package has been installed, so it has a version number
        # from pyproject.toml
        version = importlib.metadata.version(__package__ or __name__)
    except importlib.metadata.PackageNotFoundError:
        # package hasn't been installed, so set version to "dev"
        version = "dev"

    return version


class SmashedWarnings:
    _WARNINGS = bool(os.environ.get("SMASHED_WARNINGS", True))

    @classmethod
    def toggle(cls, value: Optional[bool] = None):
        if value is None:
            value = not cls._WARNINGS
        cls._WARNINGS = value

    @classmethod
    def _warn(
        cls: Type["SmashedWarnings"],
        message: str,
        category: Type[Warning],
        stacklevel: int = 2,
    ):
        if cls._WARNINGS:
            warnings.warn(message, category, stacklevel=stacklevel)

    @classmethod
    def deprecation(cls, message: str):
        cls._warn(message, RuntimeWarning)

    @classmethod
    def precedence(cls, message: str):
        cls._warn(message, RuntimeWarning)


def get_cache_dir(custom_cache_dir: Optional[Union[Path, str]] = None) -> Path:
    """Get the path to the cache directory."""

    if custom_cache_dir is not None:
        cache_dir = (
            Path(custom_cache_dir) / "allenai" / "smashed" / get_version()
        )
    else:
        cache_dir = Path(
            platformdirs.user_cache_dir(
                appname="smashed", appauthor="allenai", version=get_version()
            )
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def int_from_bytes(b: bytes) -> int:
    """Convert a byte string to an integer."""
    return int.from_bytes(b, byteorder="big")


def bytes_from_int(i: int) -> bytes:
    """Convert an integer to a byte string."""
    return i.to_bytes((i.bit_length() + 7) // 8, byteorder="big")


def make_pipeline(
    first_mapper: M,
    *rest_mappers: Union["SingleBaseMapper", "BatchedBaseMapper"]
) -> M:
    """Make a pipeline of mappers."""
    for mapper in rest_mappers:
        first_mapper = first_mapper.chain(mapper)
    return first_mapper
