import json

from smashed.base.mapper import SingleBaseMapper
from smashed.base.types import TransformElementType


class ReadJsonFilesMapper(SingleBaseMapper):
    def __init__(self, file_paths_field: str) -> None:
        """A mapper that reads json files from filepaths stored in
        `file_paths_field` and populates dataset with their content.
        Json keys are added as new fields to the dataset.
        This mapper is espically usefull when there are a lot of files
        and we want to read them in parallel (see `num_proc` attribute
        in the Huggingface's Dataset).

        Args:
            file_paths_field (str): A field in a dataset where the filepaths
                are stored.
        """
        super().__init__(input_fields=[file_paths_field], output_fields=None)

    def transform(self, data: TransformElementType) -> TransformElementType:
        file_path = data[self.input_fields[0]]
        with open(file_path, "r") as f:
            new_data_dict = json.load(f)
        data.update(new_data_dict)
        return new_data_dict


class SaveAsJsonFilesMapper(SingleBaseMapper):
    def __init__(self, file_paths_field: str) -> None:
        """A mapper that serializes dataset examples as separate
        json files whos pathes are provided in the `file_paths_field`.
        Does not modify the dataset itself.

        Args:
            file_paths_field (str): A field in a dataset where the filepaths
                are stored.
        """
        super().__init__(input_fields=[file_paths_field], output_fields=None)

    def transform(self, data: TransformElementType) -> TransformElementType:
        file_path = data[self.input_fields[0]]
        with open(file_path, "w") as f:
            json.dump(data, f)
        return data
