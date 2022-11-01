from typing import List

import numpy as np
from typing_extensions import Annotated

from ..base import SingleBaseMapper, TransformElementType


class IndicesToMaskMapper(SingleBaseMapper):
    """Converts a field containing a one or a list of indices to a mask."""

    def __init__(
        self,
        mask_field_name: str,
        reference_field_name: str,
        locations_field_name: str,
        mask_off_value: int = 0,
        mask_fill_value: int = 1,
    ):
        """
        Args:
            mask_field_name (str): Name of the field to create to contain
                the mask values.
            reference_field_name (str): Name of the field to be used to
                determine the size of the mask.
            locations_field_name (str): Name of the field containing the
                indices to be used to fill the mask.
            mask_off_value (int, optional): Value to use for the mask when
                the location is not in the list of indices. Defaults to 0.
            mask_fill_value (int, optional): Value to use for the mask when
                the location is in the list of indices. Defaults to 1.
        """

        self.mask_field_name = mask_field_name
        self.reference_field_name = reference_field_name
        self.locations_field_name = locations_field_name

        self.mask_off_value = mask_off_value
        self.mask_fill_value = mask_fill_value

        super().__init__(
            input_fields=(locations_field_name, reference_field_name),
            output_fields=(mask_field_name,),
        )

    def transform(self, data: TransformElementType) -> TransformElementType:
        locs = (
            [data[self.locations_field_name]]
            if isinstance(data[self.locations_field_name], int)
            else data[self.locations_field_name]
        )

        mask = np.full(
            len(data[self.reference_field_name]),
            self.mask_off_value,
        )
        mask[locs] = self.mask_fill_value

        return {self.mask_field_name: mask.tolist()}


class RangeToMaskMapper(IndicesToMaskMapper):
    """Converts a field containing one or more ranges of indices to a mask."""

    def transform(self, data: TransformElementType) -> TransformElementType:
        if len(data[self.locations_field_name]) == 0:
            # in case of empty ranges, return a mask of zeros
            empty_mask = [0] * len(data[self.reference_field_name])
            return {self.mask_field_name: empty_mask}

        if isinstance(data[self.locations_field_name][0], list):
            # this means we have more than one start/end pair
            locs = data[self.locations_field_name]
        else:
            # this means the field is a single [start, end] pair;
            # we need to make it a list of pairs
            locs = [data[self.locations_field_name]]

        mask = np.full(
            len(data[self.reference_field_name]),
            self.mask_off_value,
            dtype=np.int32,
        )
        for start, end in locs:
            mask[start:end] = self.mask_fill_value

        return {self.mask_field_name: mask.tolist()}


class MaskToIndicesMapper(SingleBaseMapper):
    """Converts a field with a mask to a list of indices."""

    def __init__(
        self,
        mask_field_name: str,
        locations_field_name: str,
        mask_off_value: int = 0,
        mask_fill_value: int = 1,
        enforce_single_location: bool = False,
    ) -> None:
        """
        Args:
            mask_field_name (str): Name of the field containing the mask
                values.
            locations_field_name (str): Name of the field to create to
                contain the indices.
            mask_off_value (int, optional): Value used in the mask when
                the location is not in the list of indices. Defaults to 0.
            mask_fill_value (int, optional): Value used in the mask when
                the location is in the list of indices. Defaults to 1.
            enforce_single_location (bool, optional): If True, the mapper
                will raise an error if the mask contains more than one
                location. Defaults to False.
        """
        self.mask_field_name = mask_field_name
        self.locations_field_name = locations_field_name
        self.enforce_single_location = enforce_single_location

        self.mask_off_value = mask_off_value
        self.mask_fill_value = mask_fill_value

        super().__init__(
            input_fields=(mask_field_name,),
            output_fields=(locations_field_name,),
        )

    def transform(self, data: TransformElementType) -> TransformElementType:
        locs = [
            i
            for i, v in enumerate(data[self.mask_field_name])
            if v == self.mask_fill_value
        ]

        if self.enforce_single_location and len(locs) != 1:
            raise ValueError(
                f"Expected exactly one location for mask field "
                f"'{self.mask_field_name}' but got {len(locs)}"
            )

        data[self.locations_field_name] = (
            locs[0] if self.enforce_single_location else locs
        )
        return data


class MaskToRangeMapper(MaskToIndicesMapper):
    """Converts a field with a mask to one or more of ranges of indices."""

    @staticmethod
    def _find_consecutive(
        data: np.ndarray, step_size: int = 1
    ) -> List[Annotated[List[int], 2]]:
        """Adapted from https://stackoverflow.com/a/7353335/938048"""
        splits = np.split(data, np.where(np.diff(data) != step_size)[0] + 1)
        return [[split[0], split[-1] + 1] for split in splits]

    def transform(self, data: TransformElementType) -> TransformElementType:
        # this is the mask we received as input; it's a mix of
        # self.mask_off_value and self.mask_fill_value; we turn it into
        # a numpy array for easier manipulation
        mask = np.array(data[self.mask_field_name])

        # locs contain all places where the mask is self.mask_fill_value;
        # some of these may be part of a contiguous range
        locs, *_ = (mask == self.mask_fill_value).nonzero()

        # pos contains the start and end indices of each contiguous range
        # in locs
        all_pos = self._find_consecutive(locs)

        if self.enforce_single_location and len(all_pos) != 1:
            raise ValueError(
                f"Expected exactly one location for mask field "
                f"'{self.mask_field_name}' but got {len(all_pos)}"
            )

        pos = all_pos[0] if self.enforce_single_location else all_pos
        return {self.locations_field_name: pos}
