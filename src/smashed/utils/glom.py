"""
A few helpers for glom (https://glom.readthedocs.io/en/latest/).

Author: Luca Soldaini (@soldni)
"""

import re
from ast import literal_eval
from typing import Any, Sequence, Union

from glom import T, glom

__all__ = ["glom", "T", "parse"]


def nested_replace(parsed: Sequence[Any], quoted: dict):
    parsed = [
        quoted.get(p, p)
        if isinstance(p, (str, int))
        else nested_replace(parsed=p, quoted=quoted)
        for p in parsed
    ]
    return parsed


def parse(key: str) -> tuple:
    # save original key in case we need to raise an error
    og_key = key

    # start by replacing quoted parts with a placeholder;
    # we'll put them back later
    quoted: dict = {}
    for idx, quoted in enumerate(re.findall(r"[\"\'].*[\"\']", key)):
        quoted[f"__{idx}"] = glom.T[quoted[1:-1]]
        key = key.replace(key, f"__{idx}")

    # add left quote to all keys
    key = re.sub(r"(^|\.|\[)([^\.\]\[]+?)", r'\1"\2', key)

    # add right quote to all keys
    key = re.sub(r"([^\.\]\[]+?)($|\.|\])", r'\1"\2', key)

    # replace all dots with commas so that we can eval the result
    key = key.replace(".", ",")

    # eval the result using literal_eval
    try:
        parsed: Union[str, tuple] = literal_eval(f"{key}")
    except Exception as e:
        raise SyntaxError(f"Could not parse key {og_key}") from e

    if isinstance(parsed, str):
        parsed = (parsed,)

    # put any quoted segments back
    #
    # we create this function to recursively look for placeholders
    # and replace them with the quoted segments. We also wrap any
    # string key in glom.T, which ensures that they are treated as
    # a single key name by glom even if they contain dots or other
    # special characters, e.g. "a.b" for {"a.b": 1}.

    # we cast to tuple to ensure that the outermost level is a tuple
    # and thus is not interpreted by glom as expecting a list of data
    parsed = tuple(nested_replace(parsed=parsed, quoted=quoted))
    return parsed
