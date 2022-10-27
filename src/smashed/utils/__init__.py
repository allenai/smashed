import importlib.metadata
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Type, TypeVar, Union

import platformdirs

from .caching import get_cache_dir
from .convert import bytes_from_int, int_from_bytes
from .version import get_name, get_name_and_version, get_version
from .warnings import SmashedWarnings

__all__ = [
    "bytes_from_int",
    "get_cache_dir",
    "get_name_and_version",
    "get_name",
    "get_version",
    "int_from_bytes",
    "SmashedWarnings",
]
