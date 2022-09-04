from ...base.mappers import SingleBaseMapper
from ...base.types import TransformElementType


class OneVsOtherAnnotatorMapper(SingleBaseMapper):
    def __init__(
        self,
        input_field: str = "annotations",
        label_field: str = "labels",
        preds_field: str = "preds",
        position: int = 0,
    ) -> None:
        super().__init__(
            input_fields=[input_field],
            output_fields=[label_field, preds_field],
        )
        self.position = position

    def transform(self, data: TransformElementType) -> TransformElementType:
        input_field_name, *_ = self.input_fields
        label_field_name, preds_field_name, *_ = self.output_fields

        if len(data[input_field_name]) < 2:
            raise ValueError(
                f"Expected at least 2 annotations, "
                f"got {len(data[input_field_name])}"
            )

        other_labels = [
            v
            for i, v in enumerate(data[input_field_name])
            if i != self.position
        ]

        return {
            preds_field_name: data[input_field_name][self.position],
            label_field_name: sum(other_labels) / len(other_labels),
        }
