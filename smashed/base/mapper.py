from abc import ABC, abstractmethod
from typing import \
    Any, Dict, Iterable, Optional, Sequence, Tuple, Type, overload

from .dataset import D
from .types import \
    TransformBatchType, TransformElementType, Features, FeatureType


class BaseMapper(ABC):
    """An abstract implementation of a Mapper; must be subclassed.

    A mapper is an object that transforms samples in a dataset; it can either
    produce one transformed sample per input sample (1-to-1 mapping,
    self.batched = False) or produce n samples per input sample (1-to-n mapping,
    self.batched = True).

    When creating a mapper from this base class, one need to assign three
    attributes:
    - self.input_fields: a list of the names of the input fields that are
        required to be present in the dataset for the mapper to work.
    - self.output_fields: a list of the names of the output fields that are
        produced by the mapper.
    - self.batched: a boolean indicating whether the mapper produces one
        transformed sample per input sample (False) or n samples per input
        sample (True).

    Beside the above attributes, the mapper also needs to implement the
    transform method, which either receives a single sample (if
    self.batched = False) or a list of samples (if self.batched = True).

    Optionally, it can also implement the cast_columns method, which returns
    a dictionary of new types for some (or all) of the input/output columns.
    """

    __batched__: Optional[bool] = None
    __input_fields__: Optional[Sequence[str]] = None
    __output_fields__: Optional[Sequence[str]] = None

    @property
    def batched(self) -> bool:
        if self.__batched__ is None:
            raise NotImplementedError('Mapper subclass must set batched')
        return self.__batched__

    @batched.setter
    def batched(self, value: bool):
        self.__batched__ = bool(value)

    @property
    def input_fields(self) -> Sequence[str]:
        if self.__input_fields__ is None:
            raise NotImplementedError('Mapper subclass must set input_fields')
        return tuple(str(e) for e in self.__input_fields__)

    @input_fields.setter
    def input_fields(self, input_fields: Sequence[str]):
        self.__input_fields__ = tuple(str(e) for e in input_fields)

    @property
    def output_fields(self) -> Sequence[str]:
        if self.__output_fields__ is None:
            raise NotImplementedError('Mapper subclass must set output_fields')
        return tuple(str(e) for e in self.__output_fields__)

    @output_fields.setter
    def output_fields(self, output_fields: Sequence[str]):
        self.__output_fields__ = tuple(str(e) for e in output_fields)

    def verify_output(self, data: Dict[str, Any]):
        for field in self.output_fields:
            if field not in data:
                raise ValueError(f'Field {field} not found in returned data')

    @classmethod
    def chain(cls: Type['BaseMapper'],
              dataset: D,
              mappers: Sequence['BaseMapper'],
              **map_kwargs: Any) -> D:
        for mapper in mappers:
            dataset = mapper.map(dataset, **map_kwargs)
        return dataset

    def map(self: 'BaseMapper',
            dataset: D,
            **map_kwargs: Any) -> D:
        for field in self.input_fields:
            if field not in dataset.features:
                msg = f'Field {field} not found in input dataset {dataset}'
                raise ValueError(msg)

        map_kwargs = {'batched': self.batched, **map_kwargs}

        if self.batched:
            dataset = dataset.map(function=self.batch_transform, **map_kwargs)
        else:
            dataset = dataset.map(function=self.transform, **map_kwargs)

        # We optionally cast some (or all!) of the output columns
        # to a new format. By default, no casting occurs.
        to_cast_features = self.cast_columns(dataset.features)
        for feat_name, feat_type in to_cast_features.items():
            dataset = dataset.cast_column(feat_name, feat_type)

        for field in self.output_fields:
            if field not in dataset.features:
                raise ValueError(f'Field {field} not found in returned data')

        return dataset

    def batch_transform(self: 'BaseMapper',
                        data: TransformBatchType) -> TransformBatchType:
        keys = [k for k in data.keys()]

        def _index_fn(t: Tuple[str, Any]) -> int:
            k, _ = t
            return keys.index(k)

        to_transform_iterable = (
            dict(zip(keys, sample)) for sample in
            zip(*(v for _, v in sorted(data.items(), key=_index_fn)))
        )
        transformed_batch = {}
        for transformed_sample in self.transform(to_transform_iterable):
            for k, v in transformed_sample.items():
                transformed_batch.setdefault(k, []).append(v)

        return transformed_batch

    @abstractmethod
    @overload
    def transform(
        self: 'BaseMapper',
        data: TransformElementType
    ) -> TransformElementType:
        ...

    @abstractmethod
    @overload
    def transform(
        self: 'BaseMapper',
        data: Iterable[TransformElementType]
    ) -> Iterable[TransformElementType]:
        ...

    @abstractmethod
    def transform(self, data):
        raise NotImplementedError('Mapper subclass must implement transform')

    def cast_columns(self, features: Features) -> Dict[str, FeatureType]:
        return {}
