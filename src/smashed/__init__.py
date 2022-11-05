from . import base, contrib, mappers, recipes, utils

pipeline = base.make_pipeline

__version__ = utils.get_version()

__all__ = [
    "base",
    "contrib",
    "mappers",
    "recipes",
    "pipeline",
    "utils",
]
