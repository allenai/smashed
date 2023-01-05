from functools import cached_property
from typing import Dict, Union

import glom
from necessary import necessary

from ..base import DataRowView, SingleBaseMapper, TransformElementType

with necessary("datasets", soft=True) as DATASETS_AVAILABLE:
    if DATASETS_AVAILABLE:
        try:
            from datasets.formatting.formatting import LazyRow
        except ImportError:
            # pre datasets 2.8.0
            from datasets.arrow_dataset import (
                Example as LazyRow,  # pyright: ignore
            )


class ExtendGlommerMixin:
    """A mixin that ensures that glom can work with huggingface Example
    and"""

    def __getstate__(self):
        state = super().__getstate__()  # pyright: ignore
        state["__dict__"].pop("glommer", None)
        return state

    @cached_property
    def glommer(self) -> glom.Glommer:
        glommer = glom.Glommer()

        if DATASETS_AVAILABLE:
            glommer.register(
                target_type=LazyRow,
                get=LazyRow.__getitem__,
                iter=LazyRow.__iter__,
                exact=LazyRow.__eq__,
            )

        glommer.register(
            target_type=DataRowView,
            get=DataRowView.__getitem__,
            iter=DataRowView.__iter__,
            exact=DataRowView.__eq__,
        )

        return glommer


class GlomMapper(ExtendGlommerMixin, SingleBaseMapper):
    """Uses glom to extract nested fields from a dict and create a flat dict.

    Example of glom syntax:

    dt2 = {'answers': {'text': [{'a': {'b': 2}}]}}
    spec = ('answers', 'text', [('a', 'b')])
    glom.glom(dt2, spec)
    print(glom.glom(dt2, spec))
    """

    def __init__(self, spec_fields: Dict[str, Union[str, tuple, glom.Spec]]):
        self.spec_fields = spec_fields

        super().__init__(output_fields=tuple(spec_fields.keys()))

    def transform(self, data: TransformElementType) -> TransformElementType:
        out = {}
        for key, spec in self.spec_fields.items():
            out[key] = self.glommer.glom(data, spec)
        return out
