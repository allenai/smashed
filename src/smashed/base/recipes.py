from typing import Any

from trouting import trouting

from .mappers import SingleBaseMapper


class BaseRecipe(SingleBaseMapper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @trouting
    def map(self, dataset: Any, **map_kwargs: Any) -> Any:
        """Recipes don't do anything, so we just return the dataset."""

        if self.pipeline is None:
            raise ValueError(
                "You have not chained any mappers! "
                "Recipes must have at least one mapper."
            )

        if self.pipeline:
            return self.pipeline.map(dataset, **map_kwargs)
        else:
            return dataset
