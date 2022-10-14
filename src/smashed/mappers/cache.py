import hashlib
import logging
import pickle
import shutil
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Union, cast

from necessary import necessary
from trouting import trouting

from smashed.utils import get_cache_dir

with necessary("datasets", soft=True) as HUGGINGFACE_DATASET_AVAILABLE:
    if HUGGINGFACE_DATASET_AVAILABLE or TYPE_CHECKING:
        from datasets.arrow_dataset import Dataset, Batch
        from datasets.iterable_dataset import IterableDataset
        from datasets.fingerprint import disable_caching, enable_caching

from ..base import SingleBaseMapper
from ..base.mappers import PipelineFingerprintMixIn
from .types import TransformElementType

__all__ = [
    "EndCachingMapper",
    "StartCachingMapper",
]


class DisableIntermediateCachingContext:
    """Disables intermediate caching if the dataset interface supports it"""

    def __init__(self, dataset: Any):
        self.dataset = dataset

    @trouting
    def disable_caching(self, dataset: Any) -> None:
        ...

    @trouting
    def enable_caching(self, dataset: Any) -> None:
        ...

    if HUGGINGFACE_DATASET_AVAILABLE:

        @disable_caching.add_interface(dataset=(Dataset, IterableDataset))
        def _disable_caching_hf(self, dataset: Any) -> None:
            disable_caching()

        @enable_caching.add_interface(dataset=(Dataset, IterableDataset))
        def _enable_caching_hf(self, dataset: Any) -> None:
            enable_caching()

    def __enter__(self) -> "DisableIntermediateCachingContext":
        self.disable_caching(self.dataset)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.enable_caching(self.dataset)


class CachePathContext:
    """Computes the path to a cache file for a given dataset and pipeline"""

    def __init__(
        self,
        base_dir: Union[str, Path],
        pipeline: Sequence[PipelineFingerprintMixIn],
        dataset: Any,
        n_samples_iterable_fingerprint: int = 10,
    ):
        self.base_dir = Path(base_dir)
        self.pipeline = pipeline
        self.dataset = dataset
        self.n_samples_iterable_fingerprint = n_samples_iterable_fingerprint

    def get_pipeline_fingerprint(
        self, pipeline: Sequence[PipelineFingerprintMixIn]
    ) -> str:
        h = hashlib.sha1()
        for mapper in pipeline:
            h.update(mapper.fingerprint.encode("utf-8"))
        return h.hexdigest()

    @trouting
    def get_dataset_fingerprint(self, dataset: Any) -> str:
        raise ValueError(
            f"I do not how to hash a dataset of type {type(dataset)}; "
            f"interface not implemented"
        )

    @get_dataset_fingerprint.add_interface(dataset=list)
    def get_dataset_fingerprint_list(
        self, dataset: List[TransformElementType]
    ) -> str:
        """The hash of a list of TransformElementTypes is the hash of the
        each single element in the list."""

        def _get_sample_hash(h, sample: TransformElementType):
            h.update(pickle.dumps(sample))
            return h

        return reduce(_get_sample_hash, dataset, hashlib.sha1()).hexdigest()

    if HUGGINGFACE_DATASET_AVAILABLE:

        @get_dataset_fingerprint.add_interface(dataset=Dataset)
        def get_dataset_fingerprint_hf_dataset(self, dataset: Dataset) -> str:
            """Mapping datasets in huggingface naturally have a fingerprint."""
            return dataset._fingerprint

        @get_dataset_fingerprint.add_interface(dataset=IterableDataset)
        def get_dataset_fingerprint_hf_iterable(
            self, dataset: IterableDataset
        ) -> str:
            """For iterable dataset, the fingerprint derived from info, split
            names, and a sample of the top n elements."""
            h = hashlib.sha1()
            h.update(
                pickle.dumps(
                    {
                        "info": dataset.info,
                        "split": dataset.split,
                        "features": dataset.features,
                        "sample": dataset._head(
                            n=self.n_samples_iterable_fingerprint
                        ),
                    }
                )
            )
            return h.hexdigest()

        @get_dataset_fingerprint.add_interface(dataset=Batch)
        def get_dataset_fingerprint_hf_batch(self, dataset: Batch) -> str:
            raise ValueError(
                "Cannot cache a Batch of a HuggingFace Dataset; please "
                "cache at the Dataset level instead."
            )

    def get_cache_path(self) -> Path:
        """Returns the full path to the cache file."""
        return (
            self.base_dir
            / self.get_dataset_fingerprint(self.dataset)
            / self.get_pipeline_fingerprint(self.pipeline)
        )

    def __enter__(self) -> Path:
        path = self.get_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            if (path := self.__enter__()).exists() and path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


class EndCachingMapper(SingleBaseMapper):
    """A mapper that indicates the end of a caching pipeline. Dataset
    received by this mapper will be cached to disk. Must be paired with a
    StartCachingMapper."""

    __slots__ = ("cache_path",)

    cache_path: Union[Path, None]

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.cache_path = None
        super().__init__()

    @trouting
    def save_cache(self, dataset: Any, path: Path) -> None:
        raise ValueError(
            "I do not how to save a dataset of type "
            f"{type(dataset)}; interface not implemented"
        )

    @save_cache.add_interface(dataset=list)
    def _save_list(
        self, dataset: List[TransformElementType], path: Path
    ) -> None:
        with open(path, "wb") as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    if HUGGINGFACE_DATASET_AVAILABLE:

        @save_cache.add_interface(dataset=Dataset)
        def _save_hf(self, dataset: Dataset, path: Path):
            dataset.save_to_disk(str(path))

        @save_cache.add_interface(dataset=IterableDataset)
        def _save_hf_it(self, dataset: IterableDataset, path: Path):
            raise NotImplementedError(
                "Saving an IterableDataset is not implemented yet"
            )

        @save_cache.add_interface(dataset=Batch)
        def _save_hf_batch(self, dataset: Dataset, path: Path):
            raise ValueError(
                "Cannot cache a Batch of a HuggingFace Dataset; please "
                "cache at the Dataset level instead."
            )

    def map(self, dataset: Any, **map_kwargs: Any) -> Any:
        if self.cache_path is None:
            raise ValueError(
                "The cache path is not set. Did you forget to add a "
                "the StartCachingMapper to the pipeline?"
            )

        self.logger.warning(f"Saving cache to {self.cache_path}")
        self.save_cache(dataset, self.cache_path)
        return (
            self.pipeline.map(dataset, **map_kwargs)
            if self.pipeline is not None
            else dataset
        )

    def transform(self, data: TransformElementType) -> TransformElementType:
        return data


class StartCachingMapper(SingleBaseMapper):
    """A mapper to indicate the position from which caching should start.
    Must be paired with an EndCachingMapper."""

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """
        Args:
            cache_dir (Optional[Union[str, Path]], optional): The directory
                where the cache should be stored. If not provided, library
                `platformdirs` will be used to determine the cache directory.

        """
        self.logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.cache_dir = get_cache_dir(cache_dir)
        super().__init__()

    def _get_pipeline_to_cache(self) -> Sequence[PipelineFingerprintMixIn]:
        current: PipelineFingerprintMixIn = self
        pipeline: List[PipelineFingerprintMixIn] = []

        # traverse the pipeline till the end
        while current.pipeline is not None:
            pipeline.append(current.pipeline)

            if isinstance(current.pipeline, EndCachingMapper):
                return pipeline
            current = current.pipeline

        raise ValueError(
            "You created a pipeline with a StartCachingMapper, but no "
            "EndCachingMapper. Please add one at the location you want "
            "caching to end."
        )

    @trouting
    def load_cache(self, path: Path, dataset: Any) -> Any:
        raise ValueError(
            f"I do not how to load a dataset of type {type(dataset)}; "
            f"interface not implemented."
        )

    @load_cache.add_interface(dataset=list)
    def _load_list(
        self, path: Path, dataset: List[TransformElementType]
    ) -> List:
        with open(path, "rb") as f:
            return pickle.load(f)

    if HUGGINGFACE_DATASET_AVAILABLE:

        @load_cache.add_interface(dataset=(IterableDataset, Dataset, Batch))
        def _load_hf(
            self,
            path: Path,
            dataset: Union[Dataset, IterableDataset],
        ) -> Dataset:
            return Dataset.load_from_disk(str(path))

    def map(self, dataset: Any, **map_kwargs: Any) -> Any:

        *pipeline, end_cache_mapper = self._get_pipeline_to_cache()

        # last element is always a EndCachingMapper, but we need to
        # explicitly type it otherwise mypy complains
        end_cache_mapper = cast(EndCachingMapper, end_cache_mapper)

        with CachePathContext(
            base_dir=self.cache_dir, dataset=dataset, pipeline=pipeline
        ) as cache_path, DisableIntermediateCachingContext(dataset):

            # we add the path to the cache in case we need to save the output
            end_cache_mapper.cache_path = cache_path

            if cache_path.exists():
                self.logger.warning(f"Loading cache from {cache_path}")

                # load the cache
                dataset = self.load_cache(path=cache_path, dataset=dataset)

                # continue the pipeline (or exit if there is no pipeline)
                return (
                    end_cache_mapper.pipeline.map(dataset, **map_kwargs)
                    if end_cache_mapper.pipeline is not None
                    else dataset
                )
            else:
                # keep going with the pipeline, the end cache mapper will
                # take care of saving the output
                return (
                    self.pipeline.map(dataset, **map_kwargs)
                    if self.pipeline is not None
                    else dataset
                )

    def transform(self, data: TransformElementType) -> TransformElementType:
        return data
