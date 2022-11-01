from functools import cached_property
from itertools import chain
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from ..base import BatchedBaseMapper, SingleBaseMapper, TransformElementType


class FlattenMapper(SingleBaseMapper):
    """Flattens a list of lists into a single list."""

    def __init__(self, field: Union[str, Sequence[str]]) -> None:
        """
        Args:
            field (str, Sequence[str]): the name of the field or multiple
                fields to flatten.
        """
        self.fields_to_flatten = [field] if isinstance(field, str) else field
        super().__init__(
            input_fields=self.fields_to_flatten,
            output_fields=self.fields_to_flatten,
        )

    def transform(self, data: TransformElementType) -> TransformElementType:
        output: Dict[str, List[Any]] = {}

        for field in self.fields_to_flatten:
            to_flatten = data[field]
            if len(to_flatten) > 0:
                while isinstance(to_flatten[0], list):
                    to_flatten = list(chain.from_iterable(to_flatten))
            output[field] = to_flatten

        return output


class UnpackingMapper(BatchedBaseMapper):
    """Unpacks some or all fields in a dataset sample.
    Useful when features contain a list of values, but you want to
    want individual values for each sample."""

    _DRP_EXTRA = "drop"
    _RPT_EXTRA = "repeat"

    fields_to_unpack: Optional[Dict[str, None]]
    fields_to_ignore: Optional[Dict[str, None]]

    def __init__(
        self,
        fields_to_unpack: Optional[Sequence[str]] = None,
        fields_to_ignore: Optional[Sequence[str]] = None,
        ignored_behavior: Optional[str] = None,
    ) -> None:
        """
        Args:
            fields_to_unpack (Optional[List[str]], optional): List of fields to
                unpack. When not None, the other fields will be repeated (if
                `ignore_behavior` is not set to "repeat") or dropped (if
                `ignore_behavior` is set to "drop"). Only one between
                `fields_to_unpack` and `fields_to_ignore` can be set. Defaults
                to None.
            fields_to_ignore (Optional[List[str]], optional): List of fields to
                ignore. When not None, the fields provided will be dropped/
                duplicated, while the others will be unpacked. Only one between
                `fields_to_unpack` and `fields_to_ignore` can be set. Defaults
                to None.
            ignore_behavior (Optional[str], optional): How to handle fields
                that are not unpacked. Can be "drop" or "repeat". Defaults to
                None. Must be set when either `fields_to_unpack` or
                `fields_to_ignore` is not None.
        """

        if fields_to_unpack is not None and fields_to_ignore is not None:
            raise ValueError(
                "Must specify only one of `fields_to_unpack` "
                "or `fields_to_ignore`"
            )

        if (
            fields_to_unpack is not None or fields_to_ignore is not None
        ) and ignored_behavior not in {self._DRP_EXTRA, self._RPT_EXTRA}:
            raise ValueError(
                f"When specifying `fields_to_unpack` or `fields_to_ignore`, "
                f"`ignore_behavior` must be one of {self._DRP_EXTRA} or "
                f"{self._RPT_EXTRA} but got {ignored_behavior} instead!"
            )

        if fields_to_unpack is not None:
            # @soldni: using `dict.fromkeys` in place of `frozenset` to avoid
            # issues with hashability: sets are not guaranteed to have the
            # same hash, which causes issues when trying to cache through
            # huggingface datasets.
            self.fields_to_unpack = dict.fromkeys(fields_to_unpack)
            self.fields_to_ignore = None

        elif fields_to_ignore is not None:
            self.fields_to_unpack = None
            # @soldni: using `dict.fromkeys` in place of `frozenset` to avoid
            # issues with hashability: sets are not guaranteed to have the
            # same hash, which causes issues when trying to cache through
            # huggingface datasets.
            self.fields_to_ignore = dict.fromkeys(fields_to_ignore)

        else:
            self.fields_to_unpack = self.fields_to_ignore = None

        self.ignore_behavior = ignored_behavior

        io_fields = (*(fields_to_unpack or []), *(fields_to_ignore or []))
        super().__init__(input_fields=io_fields, output_fields=io_fields)

    def _check_wether_to_unpack(self, field_name: str) -> bool:
        if self.fields_to_unpack is not None:
            return field_name in self.fields_to_unpack
        elif self.fields_to_ignore is not None:
            return field_name not in self.fields_to_ignore
        else:
            return True

    def transform(
        self, data: Iterable[TransformElementType]
    ) -> Iterable[TransformElementType]:

        # we get the names of fields to unpack after seeing the
        # first sample; so we set this to None for now
        all_field_names_to_unpack: Optional[List[str]] = None

        # iterate over all samples
        for packed_sample in data:

            # first, compute the names of fields to unpack
            if all_field_names_to_unpack is None:
                all_field_names_to_unpack = [
                    k
                    for k in packed_sample.keys()
                    if self._check_wether_to_unpack(k)
                ]

            if len(all_field_names_to_unpack) == 0:
                # raise an error if there's nothing to unpack,
                # which might indicate an error in setting up the mapper
                raise ValueError("No fields to unpack!")

            unpacked_samples_it: Iterable[Dict[str, Any]] = (
                # this re-attached all the field names to just the values
                # of the new unpacked samples.
                {
                    k: v
                    for k, v in zip(
                        all_field_names_to_unpack, unpacked_element_values
                    )
                }
                # this zip goes from list of features, each containing
                # multiple elements in this packed sample, to a list of
                # list of features in a element.
                for unpacked_element_values in zip(
                    *(packed_sample[k] for k in all_field_names_to_unpack)
                )
            )

            if self.ignore_behavior == self._RPT_EXTRA:
                # duplicate the fields that have not been unpacked
                # in all new samples

                features_to_duplicate = {
                    k: v
                    for k, v in packed_sample.items()
                    if k not in all_field_names_to_unpack
                }
                unpacked_samples_it = (
                    # this duplicates the unpacked samples
                    {**unpacked_sample, **features_to_duplicate}
                    for unpacked_sample in unpacked_samples_it
                )

            yield from unpacked_samples_it


class SingleSequenceStriderMapper(BatchedBaseMapper):
    """Mapper that creates multiple sequences from a single field
    if the field is longer than the provided maximum length; an optional
    stride can be used to create overlapping sequences."""

    def __init__(
        self,
        field_to_stride: Union[str, Sequence[str]],
        max_length: int,
        stride: Optional[int] = None,
        keep_last: bool = False,
    ):
        """
        Args:
            field_to_stride (str, List[str]): Name of the field or fields to
                stride.
            max_length (int): Maximum length for each sequence; if a sequence
                is longer than this, it will be split into multiple sequences.
            stride (Optional[int], optional): Step to use when striding. If not
                provided, the stride step will be equal to `max_length`, which
                will create non-overlapping sequences. Defaults to None.
        """

        self.fields_to_stride = dict.fromkeys(
            [field_to_stride]
            if isinstance(field_to_stride, str)
            else field_to_stride
        )
        self.max_length = max_length
        self.keep_last = keep_last
        self.stride = stride or max_length

        super().__init__(
            input_fields=self.fields_to_stride,
            output_fields=self.fields_to_stride,
        )

    @cached_property
    def ref_field(self) -> str:
        return next(iter(self.fields_to_stride))

    def _transform_single(
        self,
        sample: TransformElementType,
    ) -> Iterable[TransformElementType]:
        seq_len = len(sample[self.ref_field])

        if seq_len < self.max_length:
            # data is too short
            yield sample

        tail_elements = 0 if self.keep_last else self.max_length
        for i in range(0, seq_len - tail_elements + 1, self.stride):

            new_sample = {
                name: (
                    values[i : i + self.max_length]
                    if name in self.fields_to_stride
                    else values
                )
                for name, values in sample.items()
            }
            yield new_sample

    def transform(
        self, data: Iterable[TransformElementType]
    ) -> Iterable[TransformElementType]:

        if len(self.fields_to_stride) < 1:
            # no fields to stride
            yield from data

        for sample in data:
            yield from self._transform_single(sample)


class SingleSequenceStriderMapperWithLocations(SingleSequenceStriderMapper):
    def __init__(
        self,
        field_to_stride: Union[str, Sequence[str]],
        max_length: int,
        field_with_locations: str,
        fields_replacement_map: Optional[Dict[str, Any]] = None,
        stride: Optional[int] = None,
    ):
        super().__init__(
            field_to_stride=field_to_stride,
            max_length=max_length,
            stride=stride,
        )
        self.field_with_locations = field_with_locations
        self.fields_replacement_map = fields_replacement_map or {}

        self.input_fields += (
            self.field_with_locations,
            *self.fields_replacement_map,
        )
        self.output_fields += (
            self.field_with_locations,
            *self.fields_replacement_map,
        )

    def _transform_single(
        self,
        sample: TransformElementType,
    ) -> Iterable[TransformElementType]:
        cum_len = 0
        for new_sample in super()._transform_single(sample):
            end_stride = cum_len + len(new_sample[self.ref_field])

            stride_is_in_locations = any(
                cum_len <= start < end_stride or cum_len < end <= end_stride
                for start, end in new_sample[self.field_with_locations]
            )

            if not stride_is_in_locations:
                # if the stride is not in the locations, we skip it
                new_sample.update(self.fields_replacement_map)

            cum_len = end_stride
            yield new_sample
