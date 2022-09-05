from ..mappers import *  # noqa
from ..utils import SmashedWarnings

SmashedWarnings.deprecation(
    "smashed.interfaces.simple is deprecated; "
    "import from smashed.mappers instead."
)


class Dataset(list):
    def __new__(cls, *args, **kwargs) -> list:  # type: ignore
        SmashedWarnings.deprecation(
            "smashed.interfaces.simple.Dataset is deprecated; "
            "simply use a list of dictionaries with str keys instead."
        )
        return list(*args, **kwargs)
