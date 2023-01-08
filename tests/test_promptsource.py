import unittest

from transformers.models.auto import AutoTokenizer

from smashed.mappers.promptsource import (
    FewShotJinjaMapper,
    PromptsourceMapper,
    JinjaMapper,
    SingleTransformPromptsourceMixin,
)
from smashed.recipes.promptsource import JinjaRecipe


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

    def test_promptsource_recipe(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        recipe = JinjaRecipe(
            tokenizer=AutoTokenizer.from_pretrained("bert-base-cased"),
            jinja_template="Q: {{question}}\nC: {{context}}\nA: |||{{answer}}",
            max_source_content_length=15,
            max_target_content_length=5,
        )
        dataset = [
            {
                "question": "What is the capital of France?",
                "context": "Paris is the capital of " + ("France " * 10),
                "answer": "Paris " * 10,
            }
        ]

        mapped_dataset, *_ = recipe.map(dataset)

        self.assertEqual(
            tokenizer.decode(mapped_dataset["input_ids"]),
            (
                "Q : What is the capital of France? "
                "C : Paris is the capital of France "
                "A :"
            ),
        )

        self.assertEqual(
            tokenizer.decode(mapped_dataset["labels"]),
            "Paris Paris Paris Paris Paris",
        )

    def test_fewshot_jinja(self):
        dataset = [
            {
                'question': 'Who is Bill Gates?',
                'answer': 'Bill Gates is a billionaire.'
            },
            {
                'question': "who is john lennon?",
                'answer': 'John Lennon was a musician.'
            },
            {
                'question': 'who is john doe?',
                'answer': 'John Doe is a fictional character.'
            },
            {
                'question': 'who is goldie hawn?',
                'answer': 'Goldie Hawn is an actress.'
            },
            {
                'question': 'who is ru paul?',
                'answer': 'Ru Paul is a drag queen.'
            }
        ]
        jinja_prompt = (
            "{% for shot in __shots__ %}"
            "Q: {{shot.question}}\n"
            "A: {{shot.answer}}\n"
            "\n"
            "{% endfor %}"
            "Q: {{question}}\n"
            "A: </s>|||{{answer}}"
        )

        mapper = FewShotJinjaMapper(jinja=jinja_prompt, num_shots=2)

        mapped_dataset = mapper.map(dataset)

        self.assertEqual(len(mapped_dataset), 1)

        self.assertEqual(
            mapped_dataset[0]['source'],
            (
                "Q: Who is Bill Gates?\nA: Bill Gates is a billionaire.\n\n"
                "Q: who is john lennon?\nA: John Lennon was a musician.\n\n"
                "Q: who is john doe?\nA: </s>"
            )
        )

        self.assertEqual(
            mapped_dataset[0]['target'],
            "John Doe is a fictional character.",
        )

        mapper = FewShotJinjaMapper(jinja=jinja_prompt, num_shots=0)

        mapped_dataset = mapper.map(dataset)

        self.assertEqual(len(mapped_dataset), 5)

        self.assertEqual(
            mapped_dataset[0]['source'],
            "Q: Who is Bill Gates?\nA: </s>"
        )

        self.assertEqual(
            mapped_dataset[0]['target'],
            "Bill Gates is a billionaire.",
        )

        self.assertEqual(
            mapped_dataset[1]['source'],
            "Q: who is john lennon?\nA: </s>"
        )
        self.assertEqual(
            mapped_dataset[1]['target'],
            "John Lennon was a musician.",
        )
