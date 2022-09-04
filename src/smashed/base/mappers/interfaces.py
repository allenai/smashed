from typing import Sequence, Iterable, Any, Optional, Sequence, Set, Tuple, Union
from trouting import trouting
from necessary import necessary
from collections import abc
from itertools import chain

from ..types import (
    ListOfDictsDatasetType,
    TransformElementType,
    HuggingFaceDataset,
    HuggingFaceIterableDataset,
)
from .abstract import (
    AbstractBaseMapper,
    AbstractSingleBaseMapper,
    AbstractBatchedBaseMapper
)


HUGGINGFACE_DATASETS_AVAILABLE = necessary('datasets', soft=True)


def check_fields_list_dataset(
    provided_fields: Union[Iterable[str], None],
    expected_fields: Sequence[str],
) -> None:
    if provided_fields is None:
        return

    provided_fields_set = set(provided_fields)

    for field in expected_fields:
        if field not in provided_fields_set:
            raise ValueError(f"Field {field} not found in dataset")


def get_iterator_and_column_names_list_dataset(
    dataset: ListOfDictsDatasetType
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



def check_fields_huggingface_dataset() -> None:
    ...


class MapMethodInterfaceMixIn(AbstractBaseMapper):

    @trouting
    def map(
        self,
        dataset: Any,
        remove_columns: Optional[Any] = False,
        **map_kwargs: Any,
    ) -> Any:
        raise ValueError(
            f"I don't know how to a dataset of type {type(dataset)}"
        )

    @map.add_interface(dataset=list)
    def map_list_of_dicts(
        self,
        dataset: ListOfDictsDatasetType,
        remove_columns: Optional[Any] = False,
        **map_kwargs: Any,
    ) -> ListOfDictsDatasetType:
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
        remove_columns = bool(remove_columns)

        if isinstance(dataset, abc.Sequence):
            check_fields_list_dataset(
                provided_fields=dataset[0].keys(),
                expected_fields=self.input_fields,
            )

        if isinstance(self, AbstractBatchedBaseMapper):
            (
                dataset_it, columns_names
            ) = get_iterator_and_column_names_list_dataset(dataset)
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
            check_fields_list_dataset(
                provided_fields=transformed_dataset[0].keys(),
                expected_fields=self.output_fields,
            )

        return transformed_dataset

    if HUGGINGFACE_DATASETS_AVAILABLE:
        @map.add_interface(dataset=HuggingFaceDataset)
        def map_huggingface_dataset(
            self,
            dataset: HuggingFaceDataset,
            *_,
            **map_kwargs: Any,
        ) -> HuggingFaceDataset:
            check_fields_list_dataset(
                provided_fields=dataset.features.keys(),
                expected_fields=self.input_fields,
            )

            if isinstance(self, BatchedBaseMapper):
                transformed_dataset = dataset.map(
                    self._batch_transform, **{**map_kwargs, "batched": True}
                )
            elif isinstance(self, SingleBaseMapper):
                transformed_dataset = dataset.map(self.transform, **map_kwargs)
            else:
                raise TypeError(
                    "Mapper must inherit a SingleBaseMapper or a BatchedBaseMapper"
                )

            self.check_dataset_fields(
                provided_fields=self.get_dataset_fields(transformed_dataset),
                expected_fields=self.output_fields,
            )

            return transformed_dataset
