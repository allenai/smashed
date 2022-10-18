from typing import Iterable, Literal, Union

from ..base import BatchedBaseMapper, TransformElementType


class FixedBatchSizeMapper(BatchedBaseMapper):

    batch_size: Union[int, float]
    keep_last: bool

    def __init__(
        self: "FixedBatchSizeMapper",
        batch_size: Union[int, Literal["max"]],
        keep_last: bool = True,
    ) -> None:
        """A very simple mapper that batches samples into fixed-size batches.

        Args:
            batch_size (Union[int, Literal["max"]): The size of the batches;
            if set to "max" it will return a single batch.
            keep_last (bool, optional): Whether to keep the last batch if it
                is not full. Defaults to True.
        """
        try:
            if batch_size == "max":
                self.batch_size = float("inf")
            else:
                self.batch_size = int(batch_size)
            assert self.batch_size > 0

        except (AssertionError, ValueError):
            raise ValueError(
                "batch_size must be a positive integer or float('inf'), "
                f"not '{batch_size}' (type: {type(batch_size)})"
            )

        if not isinstance(keep_last, bool):
            raise ValueError(f"keep_last must be a boolean, not {keep_last}")

        self.keep_last = keep_last
        super().__init__()

    def transform(
        self: "FixedBatchSizeMapper", data: Iterable[TransformElementType]
    ) -> Iterable[TransformElementType]:
        accumulator = None
        counter = 0
        for sample in data:
            if accumulator is None:
                accumulator = {k: [v] for k, v in sample.items()}
            else:
                [accumulator[k].append(v) for k, v in sample.items()]
            counter += 1

            if counter == self.batch_size:
                yield accumulator
                accumulator = None
                counter = 0

        if self.keep_last and accumulator is not None:
            yield accumulator
