import os
import warnings
from typing import Optional, Type


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
