from collections import abc
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from necessary import necessary
from trouting import trouting

from ..types import TransformBatchType, TransformElementType
from .abstract import (
    AbstractBaseMapper,
    AbstractBatchedBaseMapper,
    AbstractSingleBaseMapper,
)

with necessary("datasets", soft=True) as HUGGINGFACE_DATASET_AVAILABLE:
    if HUGGINGFACE_DATASET_AVAILABLE or TYPE_CHECKING:
        from datasets.arrow_dataset import Dataset
        from datasets.iterable_dataset import IterableDataset

        HuggingFaceDataset = TypeVar(
            "HuggingFaceDataset", Dataset, IterableDataset
        )


class MapMethodInterfaceMixIn(AbstractBaseMapper):
    def _check_fields_datasets(
        self,
        provided_fields: Union[Iterable[str], None],
        expected_fields: Sequence[str],
    ) -> None:
        if provided_fields is None:
            return

        provided_fields_set = set(provided_fields)

        for field in expected_fields:
            if field not in provided_fields_set:
                raise ValueError(f"Field {field} not found in dataset")

    def _get_iterator_and_column_names_list_dataset(
        self,
        dataset: Sequence[TransformElementType],
    ) -> Tuple[Iterable[TransformElementType], Set[str]]:
        """Given an iterable dataset, return the name of the columns
        as well as an iterator over the dataset."""

        dataset_iter = iter(dataset)
        try:
            first_element = next(dataset_iter)
        except StopIteration:
            return iter([]), set()

        column_names: Set[str] = {str(e) for e in first_element.keys()}

        dataset_iter_chained: Iterable[TransformElementType] = chain(
            (first_element,), dataset_iter
        )
        return dataset_iter_chained, column_names

    def _batch_transform_huggingface_datasets(
        self, data: TransformBatchType
    ) -> TransformBatchType:
        """Unrolls a datasets.Dataset batch, which is a dictionary of
        <features, list of feature values for each sample> into a iterable
        of dictionaries that can be passed to the transform function."""

        keys = [k for k in data.keys()]

        # _index_fn ensures that, between when we unpack
        # the sequence of samples in TrasformBatchType, and when we
        # pack them into a list of dictionaries, we always get the
        # same order of features. This is important because we don't
        # want one feature value accidentally getting mapped to the
        # wrong feature name
        def _index_fn(t: Tuple[str, Any]) -> int:
            k, _ = t
            return keys.index(k)

        to_transform_iterable = (
            dict(zip(keys, sample))
            for sample in zip(
                *(v for _, v in sorted(data.items(), key=_index_fn))
            )
        )
        transformed_batch: Dict[str, List[Any]] = {}
        for transformed_sample in self.transform(to_transform_iterable):
            for k, v in transformed_sample.items():
                transformed_batch.setdefault(k, []).append(v)

        return transformed_batch

    @trouting
    def map(  # type: ignore
        self,
        dataset: Any,
        **map_kwargs: Any,
    ) -> Any:
        raise ValueError(
            f"I don't know how to a dataset of type {type(dataset)}"
        )

    @map.add_interface(dataset=list)
    def map_list_of_dicts(
        self,
        dataset: Sequence[TransformElementType],
        **map_kwargs: Any,
    ) -> Sequence[TransformElementType]:
        """Transform a dataset by applying this mapper's transform method.

        Args:
            dataset (DatasetType): The dataset to transform.
            remove_columns (Optional[bool], optional): If True, remove discard
                any columns that are in the input dataset, but are not returned
                by the transform method. Defaults to False.
            map_kwargs (Any, optional): Additional keyword arguments to pass to
                the transform method. By default, this is empty; other
                implementations may use this.
        """

        # explicitly casting to a boolean since this is all that is
        # supported by the simple mapper.
        # TODO[lucas]: maybe support specifying which fields to keep?
        remove_columns = bool(map_kwargs.pop("remove_columns", False))

        if isinstance(dataset, abc.Sequence):
            self._check_fields_datasets(
                provided_fields=dataset[0].keys(),
                expected_fields=self.input_fields,
            )

        if isinstance(self, AbstractBatchedBaseMapper):
            (
                dataset_it,
                columns_names,
            ) = self._get_iterator_and_column_names_list_dataset(dataset)
            transformed_dataset_it = self.transform(dataset_it)

            if remove_columns:
                transformed_dataset_it = (
                    {k: v for k, v in elem.items() if k in columns_names}
                    for elem in transformed_dataset_it
                )
            transformed_dataset = list(transformed_dataset_it)

        elif isinstance(self, AbstractSingleBaseMapper):
            if remove_columns:
                # we don't care about the original columns
                transformed_dataset = [
                    self.transform(sample) for sample in dataset
                ]
            else:
                # user wants to keep the columns, so we merge the new fields
                # with the old fields, while keeping the new ones if there
                # is a name conflict
                transformed_dataset = [
                    {**sample, **self.transform(sample)} for sample in dataset
                ]
        else:
            raise TypeError(
                "Mapper must inherit a SingleBaseMapper or a BatchedBaseMapper"
            )

        if isinstance(dataset, abc.Sequence):
            self._check_fields_datasets(
                provided_fields=transformed_dataset[0].keys(),
                expected_fields=self.output_fields,
            )

        return transformed_dataset

    if HUGGINGFACE_DATASET_AVAILABLE:

        @map.add_interface(dataset=(Dataset, IterableDataset))
        def map_huggingface_dataset(
            self,
            dataset: HuggingFaceDataset,
            **map_kwargs: Any,
        ) -> HuggingFaceDataset:
            self._check_fields_datasets(
                provided_fields=dataset.features.keys(),
                expected_fields=self.input_fields,
            )

            if isinstance(self, AbstractBatchedBaseMapper):
                transformed_dataset = dataset.map(
                    self._batch_transform_huggingface_datasets,
                    **{**map_kwargs, "batched": True},  # type: ignore
                )
            elif isinstance(self, AbstractSingleBaseMapper):
                transformed_dataset = dataset.map(self.transform, **map_kwargs)
            else:
                raise TypeError(
                    "Mapper but be either a SingleBaseMapper or "
                    "a BatchedBaseMapper"
                )

            self._check_fields_datasets(
                provided_fields=transformed_dataset.features.keys(),
                expected_fields=self.output_fields,
            )

            return transformed_dataset
