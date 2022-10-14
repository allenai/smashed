from itertools import chain
from typing import Any, Dict, Iterable, List, Optional

from ..base import BatchedBaseMapper, SingleBaseMapper, TransformElementType


class FlattenMapper(SingleBaseMapper):
    def __init__(self, field: str) -> None:
        super().__init__(input_fields=[field], output_fields=[field])

    def transform(self, data: TransformElementType) -> TransformElementType:
        field_name, *_ = self.input_fields

        flattened_field = data[field_name]

        if len(flattened_field) > 0:
            while isinstance(flattened_field[0], list):
                flattened_field = list(chain.from_iterable(flattened_field))

        return {field_name: flattened_field}


class UnpackingMapper(BatchedBaseMapper):
    """Unpacks some or all fields in a dataset sample.
    Useful when features contain a list of values, but you want to
    want individual values for each sample."""

    _DRP_EXTRA = "drop"
    _RPT_EXTRA = "repeat"

    def __init__(
        self,
        fields_to_unpack: Optional[List[str]] = None,
        fields_to_ignore: Optional[List[str]] = None,
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
            field_names = set(fields_to_unpack)
            self.check_unpack_fn = lambda f: f in field_names
        elif fields_to_ignore is not None:
            field_names = set(fields_to_ignore)
            self.check_unpack_fn = lambda f: f not in field_names
        else:
            field_names = None
            # unpack all!
            self.check_unpack_fn = lambda _: True

        self.ignore_behavior = ignored_behavior

        super().__init__(
            input_fields=list(field_names) if field_names else None,
            output_fields=list(field_names) if field_names else None,
        )

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
                    k for k in packed_sample.keys() if self.check_unpack_fn(k)
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
