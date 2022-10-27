from pathlib import Path
from typing import Optional, Union

import platformdirs

from .version import get_version


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
