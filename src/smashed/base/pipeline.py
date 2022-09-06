import hashlib
from functools import reduce
from itertools import chain
from typing import Any, Tuple, Type, Union

from .mappers.abstract import AbstractBaseMapper


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

        def _to_pipeline(
            mapper_or_pipeline: Union[AbstractBaseMapper, "Pipeline"]
        ) -> "Pipeline":
            if isinstance(mapper_or_pipeline, AbstractBaseMapper):
                mapper_or_pipeline = Pipeline(mapper_or_pipeline)
            return mapper_or_pipeline

        return Pipeline(
            *chain.from_iterable(
                _to_pipeline(m_or_p).mappers for m_or_p in mappers_or_pipelines
            )
        )

    def get_pipeline_fingerprint(self) -> str:
        h = hashlib.sha1()
        for mapper in self.mappers:
            h.update(mapper.fingerprint.encode("utf-8"))
        return h.hexdigest()

    def get_dataset_fingerprint(self, dataset: Any):
        ...

    def map(self: "Pipeline", dataset: Any, **map_kwargs: Any) -> Any:
        """Transform a dataset by applying this pipeline's mappers."""

        # IMPLEMENTATION FOR DEBUG PURPOSES
        # for mapper in self.mappers:
        #     dataset = mapper.map(dataset, **map_kwargs)
        #     if not dataset:
        #         breakpoint()
        # return dataset

        def _map(dataset: Any, mapper: AbstractBaseMapper) -> Any:
            return mapper.map(dataset, **map_kwargs)

        return reduce(_map, self.mappers, dataset)
