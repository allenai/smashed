import importlib
import os
import warnings
from typing import Optional, Type, Union, TYPE_CHECKING

from packaging.version import LegacyVersion, Version, parse


def requires(
    module_name: str,
    required_version: Optional[Union[str, Version, LegacyVersion]] = None,
    soft: bool = False,
    allow_type_checking: bool = True
) -> bool:
    """Function to check if a module is installed and optionally check its
    version. If `soft` is True, the function will return False if the module is
    not installed. If `soft` is False, the function will raise an ImportError.
    """
    if allow_type_checking and TYPE_CHECKING:
        return True

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        if soft:
            return False
        else:
            raise ImportError(f"{module_name} is required for this module")

    if required_version is not None:
        module_version = parse(module.__version__)

        if not isinstance(required_version, (Version, LegacyVersion)):
            required_version = parse(required_version)

        if required_version > module_version:
            if soft:
                return False
            else:
                raise ImportError(
                    f"Version {required_version} is required for module "
                    f"{module}, but you have {module_name} version "
                    f"{module_version}"
                )
    return True


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

    @classmethod
    def precedence(cls, message: str):
        cls._warn(message, RuntimeWarning)
