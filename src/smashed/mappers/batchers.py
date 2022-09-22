from typing import Iterable

from ..base import BatchedBaseMapper, TransformElementType


class FixedBatchSizeMapper(BatchedBaseMapper):
    def __init__(
        self: "FixedBatchSizeMapper",
        batch_size: int,
        keep_last: bool = True,
    ) -> None:
        """A very simple mapper that batches samples into fixed-size batches.

        Args:
            batch_size (int): The number of samples in each batch.
            keep_last (bool, optional): Whether to keep the last batch if it
                is not full. Defaults to True.
        """
        try:
            batch_size = int(batch_size)
            assert batch_size > 0
        except Exception:
            raise ValueError(
                f"batch_size must be a positive integer, not {batch_size}"
            )

        if not isinstance(keep_last, bool):
            raise ValueError(f"keep_last must be a boolean, not {keep_last}")

        self.batch_size = batch_size
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
