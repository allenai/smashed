import importlib
import os
import warnings
from typing import Optional, Type, Union

from packaging.version import LegacyVersion, Version, parse


def requires(
    module_name: str,
    required_version: Optional[Union[str, Version, LegacyVersion]] = None,
):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ImportError(f"{module_name} is required for this module")

    if required_version is not None:
        module_version = parse(module.__version__)

        if not isinstance(required_version, (Version, LegacyVersion)):
            required_version = parse(required_version)

        if required_version > module_version:
            raise ImportError(
                f"Version {required_version} is required for module {module}, "
                f"but you have {module_name} version {module_version}"
            )


class SmashedWarnings:
    _WARNINGS = os.environ.get("SMASHED_WARNINGS", False) is True

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
        cls._warn(message, DeprecationWarning)
