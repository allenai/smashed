from functools import reduce
from itertools import chain
from typing import Any, Tuple, Type, Union

from .mapper import AbstractBaseMapper
from .types import DatasetType


class Pipeline:
    """A pipeline is a sequence of mappers that are applied to a dataset.

    Pipelines can be created by chaining two mappers together or by chaining
    a pipeline with another pipeline or mapper. Use operators << and >> to
    chain mappers or pipelines, or call the "chain" method on the class itself.

    To execute a pipeline, call the pipeline with a dataset as the first
    argument; any additional arguments are passed to the map method of each
    mapper in the pipeline.
    """

    mappers: Tuple[AbstractBaseMapper, ...]

    def __init__(self, *mappers: AbstractBaseMapper) -> None:
        self.mappers = mappers

    def __repr__(self: "Pipeline") -> str:
        mappers_it = (repr(m) for m in self.mappers)
        return f'Pipeline({" -> ".join(mappers_it)})'

    def __str__(self: "Pipeline") -> str:
        mappers_it = (str(m) for m in self.mappers)
        return f'Pipeline({" -> ".join(mappers_it)})'

    def __lshift__(
        self: "Pipeline", other: Union[AbstractBaseMapper, "Pipeline"]
    ) -> "Pipeline":
        return self.chain(other, self)

    def __rshift__(
        self: "Pipeline", other: Union[AbstractBaseMapper, "Pipeline"]
    ) -> "Pipeline":
        return self.chain(self, other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pipeline):
            return False
        if len(self.mappers) != len(other.mappers):
            return False
        for my_mapper, theirs_mapper in zip(self.mappers, other.mappers):
            if my_mapper != theirs_mapper:
                return False
        return True

    @classmethod
    def chain(
        cls: Type["Pipeline"],
        *mappers_or_pipelines: Union[AbstractBaseMapper, "Pipeline"],
    ) -> "Pipeline":
        """Create a new pipeline by chaining two mappers/pipelines together."""

        def _to_pip(
            mapper_or_pipeline: Union[AbstractBaseMapper, "Pipeline"]
        ) -> "Pipeline":
            if isinstance(mapper_or_pipeline, AbstractBaseMapper):
                mapper_or_pipeline = Pipeline(mapper_or_pipeline)
            return mapper_or_pipeline

        return Pipeline(
            *chain.from_iterable(
                _to_pip(m_or_p).mappers for m_or_p in mappers_or_pipelines
            )
        )

    def map(
        self: "Pipeline", dataset: DatasetType, **map_kwargs: Any
    ) -> DatasetType:
        """Transform a dataset by applying this pipeline's mappers."""

        def _map(dataset: DatasetType, mapper: AbstractBaseMapper):
            return mapper.map(dataset, **map_kwargs)

        return reduce(_map, self.mappers, dataset)

    def transform(self: "Pipeline", dataset: DatasetType) -> DatasetType:
        """Transform a dataset by applying this pipeline's mappers."""
        return self.map(dataset)
