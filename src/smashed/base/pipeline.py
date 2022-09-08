from pathlib import Path
from functools import reduce
from typing import Any, Iterator, Optional, Sequence, Tuple, Union

from .mappers.abstract import AbstractBaseMapper
from .cache import PipelineCacheUtils


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

    def __init__(
        self,
        *mappers_or_pipelines: Union[
            AbstractBaseMapper,
            Sequence[AbstractBaseMapper],
            "Pipeline",
        ],
    ) -> None:
        """
        Create a pipeline from a sequence of mappers.

        Args:
            mappers (Union[AbstractBaseMapper, Sequence[AbstractBaseMapper],
                "Pipeline"]): A combination of single mappers, sequences of
                mappers, and pipelines. Nested iterables are flattened.
        """
        self.mappers = tuple(
            m
            for m_or_p in mappers_or_pipelines
            for m in (
                # if it is a single mapper, we need to wrap it in a single
                # element list; otherwise, we can just iterate over it
                [m_or_p]
                if isinstance(m_or_p, AbstractBaseMapper)
                else iter(m_or_p)
            )
        )

    def __repr__(self: "Pipeline") -> str:
        mappers_it = (repr(m) for m in self.mappers)
        return f'Pipeline({" -> ".join(mappers_it)})'

    def __str__(self: "Pipeline") -> str:
        mappers_it = (str(m) for m in self.mappers)
        return f'Pipeline({" -> ".join(mappers_it)})'

    def __lshift__(
        self: "Pipeline", other: Union[AbstractBaseMapper, "Pipeline"]
    ) -> "Pipeline":
        return type(self)(other, self)

    def __rshift__(
        self: "Pipeline", other: Union[AbstractBaseMapper, "Pipeline"]
    ) -> "Pipeline":
        return type(self)(self, other)

    def __len__(self: "Pipeline") -> int:
        return len(self.mappers)

    def __getitem__(self: "Pipeline", index: int) -> AbstractBaseMapper:
        return self.mappers[index]

    def __iter__(self: "Pipeline") -> Iterator[AbstractBaseMapper]:
        # This is not strictly necessary, but mypy fails to recognize an
        # object as iterable if it only implements __getitem__. This is a
        # long-standing issue: https://github.com/python/mypy/issues/2220
        # but no one seems to be wanting to fix it.
        #
        # Returning an iterator here is a workaround, and should have no
        # to very little performance impact.
        return iter(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pipeline):
            return False
        if len(self.mappers) != len(other.mappers):
            return False
        for my_mapper, theirs_mapper in zip(self.mappers, other.mappers):
            if my_mapper != theirs_mapper:
                return False
        return True

    def map(
        self: "Pipeline",
        dataset: Any,
        use_cache: Optional[Union[str, Path, bool]] = False,
        **map_kwargs: Any
    ) -> Any:
        """Transform a dataset by applying this pipeline's mappers."""

        # this function is to be used in the reduce operation below
        # to apply each mapper in the pipeline to the dataset
        def _apply_mapper(dataset: Any, mapper: AbstractBaseMapper) -> Any:
            return mapper.map(dataset, **map_kwargs)

        # cache path will be None if use_cache is False
        cache_path = PipelineCacheUtils.get_cache_path(
            dataset=dataset,
            pipeline=self,
            use_cache=use_cache
        )

        if cache_path and cache_path.exists():
            # load from cache
            transformed_dataset = PipelineCacheUtils.load_transformed_dataset(
                cache_path=cache_path,
                source_dataset=dataset
            )
        else:
            # compute from scratch
            with PipelineCacheUtils.no_cache_ctx(
                dataset=dataset, no_caching=bool(use_cache)
            )():
                # disable intermediate caching if we are using an end-to-end
                # cache and are dealing with huggingface datasets
                transformed_dataset = reduce(
                    _apply_mapper, self.mappers, dataset
                )

            if cache_path:
                # save if we need to
                PipelineCacheUtils.save_transformed_dataset(
                    transformed_dataset=transformed_dataset,
                    cache_path=cache_path,
                )

        return transformed_dataset
