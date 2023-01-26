from datasets.load import load_dataset

import smashed.mappers as sm

dataset = load_dataset("squad", split="validation")

pipeline = (
    sm.GlomMapper(
        spec_fields={
            "question": "question",
            "context": "context",
            "answers": ("answers", "text", tuple()),
        }
    )
    >> sm.TextToWordsMapper(
        fields=["question", "context", "answers"],
        splitter="ws",
    )
    >> sm.SingleSequenceStriderMapper(
        field_to_stride=["context"],
        max_length=300,
        stride=100,
    )
    >> sm.TruncateSingleFieldMapper(
        fields_to_truncate={
            "question": 50,
            "context": 300,
            "answers": 400,
        }
    )
    >> sm.WordsToTextMapper(
        fields=["question", "context", "answers"],
    )
    >> sm.JinjaMapper(
        jinja=(
            "Q:{{question}}\nC:{{context}}\nA: "
            "{% for answer in answers %}|||{{answer}}{% endfor %}"
        ),
        return_multiple_targets=True,
    )
)

mapped_data = pipeline.map(dataset, num_proc=4)

for i in range(10):
    print(mapped_data[i]["source"])
    print(mapped_data[i]["target"])
    print()
