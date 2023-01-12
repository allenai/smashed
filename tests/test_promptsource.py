import unittest

from smashed.mappers.promptsource import (
    FewShotJinjaMapper,
    JinjaMapper,
    PromptsourceMapper,
    SingleTransformPromptsourceMixin,
)

FEW_SHOT_DATASET = [
    {
        "question": "Who is Bill Gates?",
        "answer": "Bill Gates is a billionaire.",
    },
    {
        "question": "who is john lennon?",
        "answer": "John Lennon was a musician.",
    },
    {
        "question": "who is john doe?",
        "answer": "John Doe is a fictional character.",
    },
    {
        "question": "who is goldie hawn?",
        "answer": "Goldie Hawn is an actress.",
    },
    {
        "question": "who is ru paul?",
        "answer": "Ru Paul is a drag queen.",
    },
]

FEW_SHOT_PROMPT = (
    "{% for shot in __shots__ %}"
    "Q: {{shot.question}}\n"
    "A: {{shot.answer}}\n"
    "\n"
    "{% endfor %}"
    "Q: {{question}}\n"
    "A: </s>|||{{answer}}"
)


class TestPromptsource(unittest.TestCase):
    def test_jinja_prompt_source_mapper(self):
        mapper = JinjaMapper(
            jinja="Q: {{question}}\nA: |||{{answers.text[0]}}"
        )
        dataset = [
            {
                "question": "What is the capital of France?",
                "context": "Paris is the capital of France.",
                "answers": {"text": ["Paris"], "answer_start": [0]},
            }
        ]
        mapped_dataset = mapper.map(dataset, remove_columns=True)
        self.assertEqual(
            mapped_dataset[0]["source"],
            "Q: What is the capital of France?\nA:",
        )
        self.assertEqual(mapped_dataset[0]["target"], "Paris")

    def test_dataset_prompt_source_mapper(self):
        mapper = PromptsourceMapper(
            dataset_name="squad",
            template_name="given_context_answer_question_variation",
        )
        dataset = [
            {
                "question": "What is the capital of France?",
                "context": "Paris is the capital of France.",
                "answers": {"text": ["Paris"], "answer_start": [0]},
            }
        ]

        mapped_dataset = mapper.map(dataset, remove_columns=True)

        self.assertEqual(len(mapped_dataset), 1)
        self.assertEqual(len(mapped_dataset[0]), 2)
        self.assertEqual(
            mapped_dataset[0]["source"],
            (
                "Paris is the capital of France.\n\n"
                "Q: What is the capital of France?\n\nA:"
            ),
        )
        self.assertEqual(mapped_dataset[0]["target"], "Paris")

        mapper2 = SingleTransformPromptsourceMixin(mapper.template)
        mapped_dataset2 = mapper2.map(dataset, remove_columns=True)
        self.assertEqual(mapped_dataset, mapped_dataset2)

    def test_fewshot_jinja(self):
        mapper = FewShotJinjaMapper(jinja=FEW_SHOT_PROMPT, num_shots=2)

        mapped_dataset = mapper.map(FEW_SHOT_DATASET)

        self.assertEqual(len(mapped_dataset), 1)

        self.assertEqual(
            mapped_dataset[0]["source"],
            (
                f"Q: {FEW_SHOT_DATASET[0]['question']}\n"
                f"A: {FEW_SHOT_DATASET[0]['answer']}\n\n"
                f"Q: {FEW_SHOT_DATASET[1]['question']}\n"
                f"A: {FEW_SHOT_DATASET[1]['answer']}\n\n"
                f"Q: {FEW_SHOT_DATASET[2]['question']}\nA: </s>"
            ),
        )

        self.assertEqual(
            mapped_dataset[0]["target"],
            FEW_SHOT_DATASET[2]["answer"],
        )

    def test_few_shot_jinja_zero_shots(self):
        mapper = FewShotJinjaMapper(jinja=FEW_SHOT_PROMPT, num_shots=0)

        mapped_dataset = mapper.map(FEW_SHOT_DATASET)

        self.assertEqual(len(mapped_dataset), 5)

        for i in range(5):
            self.assertEqual(
                mapped_dataset[i]["source"],
                f"Q: {FEW_SHOT_DATASET[i]['question']}\nA: </s>",
            )

            self.assertEqual(
                mapped_dataset[i]["target"],
                FEW_SHOT_DATASET[i]["answer"],
            )

    def test_few_shot_exception(self):
        with self.assertRaises(KeyError):
            FewShotJinjaMapper(
                jinja="Q: {{question}}\nA: {{answer}}", num_shots=2
            )

        with self.assertRaises(ValueError):
            FewShotJinjaMapper("{{ __shots__ }}", num_shots=-2)
