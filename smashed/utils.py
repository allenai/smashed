import importlib
from packaging.version import Version, LegacyVersion, parse
from typing import Optional, Union


def requires(
    module_name: str,
    required_version: Optional[Union[str, Version, LegacyVersion]] = None
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
