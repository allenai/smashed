import hashlib
import pickle
from contextlib import contextmanager
from functools import reduce
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    List,
    Optional,
    Union,
)

from necessary import necessary
from trouting import trouting

from smashed.utils import get_cache_dir

with necessary("datasets", soft=True) as HUGGINGFACE_DATASET_AVAILABLE:
    if HUGGINGFACE_DATASET_AVAILABLE or TYPE_CHECKING:
        from datasets.arrow_dataset import Dataset
        from datasets.iterable_dataset import IterableDataset
        from datasets.fingerprint import disable_caching, enable_caching

from .types import TransformElementType

if TYPE_CHECKING:
    from .pipeline import Pipeline


class PipelineCacheUtils:
    @staticmethod
    def get_pipeline_fingerprint(pipeline: "Pipeline") -> str:
        h = hashlib.sha1()
        for mapper in pipeline:
            h.update(mapper.fingerprint.encode("utf-8"))
        return h.hexdigest()

    @trouting
    @classmethod
    def no_cache_ctx(cls, dataset: Any, no_caching: bool = False) -> Callable:
        @contextmanager
        def _no_cache_ctx() -> Iterator[None]:
            try:
                yield
            finally:
                pass

        return _no_cache_ctx

    @no_cache_ctx.add_interface(dataset=Dataset)
    @classmethod
    def _no_cache_hf(cls, dataset: Any, no_caching: bool = False) -> Callable:
        @contextmanager
        def _no_cache_ctx() -> Iterator[None]:
            try:
                if no_caching:
                    disable_caching()
                yield
            finally:
                if no_caching:
                    enable_caching()

        return _no_cache_ctx

    @trouting
    @classmethod
    def get_dataset_fingerprint(cls, dataset: Any) -> str:
        raise ValueError(
            f"I do not how to hash a dataset of type {type(dataset)}; "
            f"interface not implemented"
        )

    @get_dataset_fingerprint.add_interface(dataset=list)
    @classmethod
    def get_dataset_fingerprint_list(
        cls, dataset: List[TransformElementType]
    ) -> str:
        """The hash of a list of TransformElementTypes is the hash of the
        each single element in the list."""

        def _get_sample_hash(h: hashlib._Hash, sample: TransformElementType):
            h.update(pickle.dumps(sample))
            return h

        return reduce(_get_sample_hash, dataset, hashlib.sha1()).hexdigest()

    if HUGGINGFACE_DATASET_AVAILABLE:

        @get_dataset_fingerprint.add_interface(dataset=Dataset)
        @classmethod
        def get_dataset_fingerprint_hf_dataset(cls, dataset: Dataset) -> str:
            """Mapping datasets in huggingface naturally have a fingerprint."""
            return dataset._fingerprint

        @get_dataset_fingerprint.add_interface(dataset=IterableDataset)
        @classmethod
        def get_dataset_fingerprint_hf_iterable(
            cls, dataset: IterableDataset
        ) -> str:
            """For iterable dataset, the fingerprint derived from info, split
            names, and a sample of the top 5 elements."""
            h = hashlib.sha1()
            h.update(
                pickle.dumps(
                    {
                        "info": dataset.info,
                        "split": dataset.split,
                        "features": dataset.features,
                        "sample": dataset._head(n=5),
                    }
                )
            )
            return h.hexdigest()

    @classmethod
    def get_cache_path(
        cls,
        dataset: Any,
        pipeline: "Pipeline",
        use_cache: Optional[Union[str, Path, bool]] = None,
    ) -> Union[Path, None]:
        if use_cache is None or use_cache is False:
            cache_dir = None
        else:
            cache_dir = (
                get_cache_dir(
                    # if it is a boolean, we use the default cache dir
                    use_cache
                    if not isinstance(use_cache, bool)
                    else None
                )
                / cls.get_dataset_fingerprint(dataset)
                / cls.get_pipeline_fingerprint(pipeline)
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    @trouting
    @classmethod
    def save_transformed_dataset(
        cls, transformed_dataset: Any, cache_path: Path
    ) -> None:
        raise ValueError(
            "I do not how to save a dataset of type "
            f"{type(transformed_dataset)}; interface not implemented"
        )

    @save_transformed_dataset.add_interface(transformed_dataset=list)
    @classmethod
    def _save_list(
        cls, transformed_dataset: List[TransformElementType], cache_path: Path
    ) -> None:
        with open(cache_path, "wb") as f:
            pickle.dump(
                transformed_dataset, f, protocol=pickle.HIGHEST_PROTOCOL
            )

    if HUGGINGFACE_DATASET_AVAILABLE:

        @save_transformed_dataset.add_interface(transformed_dataset=Dataset)
        @classmethod
        def _save_hf(cls, transformed_dataset: Dataset, cache_path: Path):
            transformed_dataset.save_to_disk(str(cache_path))

        @save_transformed_dataset.add_interface(
            transformed_dataset=IterableDataset
        )
        @classmethod
        def _save_hf_it(
            cls, transformed_dataset: IterableDataset, cache_path: Path
        ):
            raise NotImplementedError(
                "Saving an IterableDataset is not implemented yet"
            )

    @trouting
    @classmethod
    def load_transformed_dataset(
        cls, cache_path: Path, source_dataset: Any
    ) -> Any:
        raise ValueError(
            f"I do not how to load a dataset of type {type(source_dataset)}; "
            f"interface not implemented."
        )

    @load_transformed_dataset.add_interface(source_dataset=list)
    @classmethod
    def _load_list(
        cls, cache_path: Path, source_dataset: List[TransformElementType]
    ) -> List:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if HUGGINGFACE_DATASET_AVAILABLE:

        @load_transformed_dataset.add_interface(
            source_dataset=(IterableDataset, Dataset)
        )
        @classmethod
        def _load_hf(
            cls,
            cache_path: Path,
            source_dataset: Union[Dataset, IterableDataset],
        ) -> Dataset:
            return Dataset.load_from_disk(str(cache_path))
