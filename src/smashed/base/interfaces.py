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
from torch._utils import classproperty
from trouting import trouting

from .abstract import (
    AbstractBaseMapper,
    AbstractBatchedBaseMapper,
    AbstractSingleBaseMapper,
)
from .types import TransformBatchType, TransformElementType
from .views import DataBatchView

with necessary("datasets", soft=True) as HUGGINGFACE_DATASET_AVAILABLE:
    if HUGGINGFACE_DATASET_AVAILABLE or TYPE_CHECKING:
        from datasets.arrow_dataset import Batch, Dataset
        from datasets.iterable_dataset import IterableDataset

        HuggingFaceDataset = TypeVar(
            "HuggingFaceDataset", Dataset, IterableDataset
        )


class MapMethodInterfaceMixIn(AbstractBaseMapper):
    """Mix-in class that implements the map method for all mappers
    and various interfaces. Do not inherit from this class directly,
    but use SingleBaseMapper/BatchedBaseMapper instead."""

    @classproperty
    def always_remove_columns(cls) -> bool:
        """Whether this mapper should always remove its input columns
        from the dataset. If False, the mapper will only remove columns
        if the output columns are not a subset of the input columns."""
        return False

    def _check_fields_datasets(
        self,
        provided_fields: Union[Iterable[str], None],
        expected_fields: Sequence[str],
        reverse_membership_check: bool = False,
    ) -> None:
        """Checks whether the provided fields are a subset of the
        expected fields. If reverse_membership_check is True, checks
        whether the provided fields are a superset of the expected
        fields."""

        if provided_fields is None:
            return

        provided_fields_set = set(provided_fields)

        if not reverse_membership_check:
            for field in expected_fields:
                if field not in provided_fields_set:
                    raise ValueError(f"Field '{field}' not found in dataset")
        else:
            for field in provided_fields_set:
                if field not in expected_fields:
                    raise ValueError(
                        f"Field '{field}' not supported by mapper "
                        f"{type(self).__name__}"
                    )

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
        # the sequence of samples in TransformBatchType, and when we
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
    def map(self, dataset: Any, **map_kwargs: Any) -> Any:
        """Transform a dataset by applying this mapper's transform method.

        Args:
            dataset (DatasetType): The dataset to transform.
            map_kwargs (Any, optional): Additional keyword arguments to
                pass to control the map operation. The available options
                differ depending on the dataset. Defaults to {}.
        """

        raise ValueError(
            f"I don't know how to map a dataset of type {type(dataset)}; "
            "interface not implemented."
        )

    @map.add_interface(dataset=list)
    def _map_list_of_dicts(
        self,
        dataset: Sequence[TransformElementType],
        **map_kwargs: Any,
    ) -> Sequence[TransformElementType]:

        # explicitly casting to a boolean since this is all that is
        # supported by the simple mapper.
        # TODO[lucas]: maybe support specifying which fields to keep?
        remove_columns = (
            bool(map_kwargs.get("remove_columns", False))
            or self.always_remove_columns
        )

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

        if self.pipeline:
            return self.pipeline.map(transformed_dataset, **map_kwargs)
        else:
            return transformed_dataset

    if HUGGINGFACE_DATASET_AVAILABLE:

        @map.add_interface(dataset=(Dataset, IterableDataset))
        def _map_huggingface_dataset(
            self,
            dataset: HuggingFaceDataset,
            **map_kwargs: Any,
        ) -> HuggingFaceDataset:
            self._check_fields_datasets(
                provided_fields=dataset.features.keys(),
                expected_fields=self.input_fields,
            )

            if self.always_remove_columns:
                remove_columns = list(dataset.features.keys())
            else:
                remove_columns = map_kwargs.get("remove_columns", [])

            if isinstance(self, AbstractBatchedBaseMapper):
                transformed_dataset = dataset.map(
                    self._batch_transform_huggingface_datasets,
                    **{
                        **map_kwargs,
                        "batched": True,
                        "remove_columns": remove_columns,
                    },
                )
            elif isinstance(self, AbstractSingleBaseMapper):
                transformed_dataset = dataset.map(
                    self.transform,
                    **{**map_kwargs, "remove_columns": remove_columns},
                )
            else:
                raise TypeError(
                    "Mapper but be either a SingleBaseMapper or "
                    "a BatchedBaseMapper"
                )

            self._check_fields_datasets(
                provided_fields=transformed_dataset.features.keys(),
                expected_fields=self.output_fields,
            )

            if self.pipeline:
                return self.pipeline.map(transformed_dataset, **map_kwargs)
            else:
                return transformed_dataset

        @map.add_interface(dataset=Batch)
        def _map_huggingface_dataset_batch(
            self,
            dataset: Batch,
            **map_kwargs: Any,
        ) -> Batch:
            # explicitly casting to a boolean since this is all that is
            # supported by the simple mapper.
            # TODO[lucas]: maybe support specifying which fields to keep?
            remove_columns = (
                bool(map_kwargs.get("remove_columns", False))
                or self.always_remove_columns
            )

            dtview: DataBatchView[Batch, str, Any] = DataBatchView(dataset)

            self._check_fields_datasets(
                provided_fields=dataset.keys(),
                expected_fields=self.input_fields,
            )

            if isinstance(self, AbstractBatchedBaseMapper):
                dtview = dtview.map(self.transform)
            elif isinstance(self, AbstractSingleBaseMapper):
                dtview.update(self.transform(dtr) for dtr in dtview)
            else:
                raise TypeError(
                    "Mapper but be either a SingleBaseMapper or "
                    "a BatchedBaseMapper"
                )

            self._check_fields_datasets(
                provided_fields=dtview.keys(),
                expected_fields=self.output_fields,
            )

            for column in tuple(dtview.keys()):
                if remove_columns and column not in self.output_fields:
                    dtview.pop(column)

            return dtview.orig()
