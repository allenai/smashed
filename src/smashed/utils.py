import importlib.metadata
import os
import warnings
from pathlib import Path
from typing import Optional, Type

import platformdirs


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


def get_cache_dir() -> Path:
    """Get the path to the cache directory."""
    (
        cache_dir := Path(
            platformdirs.user_cache_dir(
                appname="smashed", appauthor="allenai", version=get_version()
            )
        )
    ).mkdir(parents=True, exist_ok=True)
    return cache_dir
