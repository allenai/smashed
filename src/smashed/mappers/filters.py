from typing import Dict, Iterable

from ..base.mappers import BatchedBaseMapper
from ..base.types import TransformElementType


class FilterMapper(BatchedBaseMapper):
    """A mapper that filters elements from a batch."""

    def __init__(self, fields_filters: Dict[str, str]):
        """
        Args:
            fields_filters (Dict[str, str]): A dictionary of fields and their
                filters. Filters are typically operators and values, e.g.
                ">= 5". We use ast.literal_eval to evaluate the filter.
        """

        self.fields_filters = fields_filters

        super().__init__(
            input_fields=list(fields_filters.keys()),
            output_fields=list(fields_filters.keys()),
        )

    def transform(
        self, data: Iterable[TransformElementType]
    ) -> Iterable[TransformElementType]:
        for batch in data:
            filter_pass = all(
                eval(f"{batch[field]} {filter_op}")
                for field, filter_op in self.fields_filters.items()
            )
            if filter_pass:
                yield batch
