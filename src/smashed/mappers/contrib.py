from ..utils import SmashedWarnings
from ..contrib import squad, sse


SmashedWarnings.deprecation("the module contrib has moved to smashed.contrib")

__all__ = ['squad', 'sse']
