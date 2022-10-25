from ..contrib import squad, sse
from ..utils import SmashedWarnings

SmashedWarnings.deprecation("the module contrib has moved to smashed.contrib")

__all__ = ["squad", "sse"]
