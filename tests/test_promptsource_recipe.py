import unittest

from transformers.models.auto import AutoTokenizer

from smashed.recipes.promptsource import JinjaRecipe

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
    {
        "question": "who is john wayne?",
        "answer": "John Wayne was an actor; he's dead!",
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
    def setUp(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            "t5-small", model_max_length=512
        )

    def test_promptsource_recipe(self):
        recipe = JinjaRecipe(
            tokenizer=self.tokenizer,
            jinja_template="Q: {{question}}\nC: {{context}}\nA: |||{{answer}}",
            # this used to be 15, but now using 'whitespace_plus' tokenizer
            # as default, which means that we have a few more tokens in the
            # non-variables part of the prompt.
            max_source_length_per_shot=18,
            max_target_length_per_shot=5,
        )
        dataset = [
            {
                "question": "What is the capital of France",
                "context": "Paris is the capital of " + ("France " * 10),
                "answer": "Paris " * 10,
            }
        ]

        mapped_dataset, *_ = recipe.map(dataset)

        self.assertEqual(
            self.tokenizer.decode(mapped_dataset["input_ids"]),
            (
                "Q: What is the capital of France "
                "C: Paris is the capital of France "
                "A:"
            ),
        )

        self.assertEqual(
            self.tokenizer.decode(mapped_dataset["labels"]),
            "Paris Paris Paris Paris Paris",
        )

    def test_few_shot_truncation(self):
        mapper = JinjaRecipe(
            tokenizer=self.tokenizer,
            jinja_template=FEW_SHOT_PROMPT,
            num_shots=2,
            max_source_length_per_shot=31,
            max_target_length_per_shot=14,
            use_words=False,
        )

        mapped_dataset = mapper.map(FEW_SHOT_DATASET)

        self.assertEqual(len(mapped_dataset), 2)

        # the total non-template length of each prompt is 20,
        # which means each should be shortened by ceil(20 / 3) = 7 chars.
        # further, we need to account for 5 characters per shot, so that's
        # a further ceil((14 * 2) / 3) = 10 chars. So from 31 the effective
        # max length should 14 for question. The answers should all get
        # truncated to 14 characters.
        #
        # The fact that the prompt is a bit different from the template
        # is totally fine: T5 removes multiple spaces, turns newlines into
        # spaces, and decoding strips the trailing spaces.
        self.assertEqual(
            self.tokenizer.decode(mapped_dataset[0]["input_ids"]),
            (
                f"Q: {FEW_SHOT_DATASET[0]['question'][:14].rstrip()} "
                f"A: {FEW_SHOT_DATASET[0]['answer'][:14].rstrip()} "
                f"Q: {FEW_SHOT_DATASET[1]['question'][:14].rstrip()} "
                f"A: {FEW_SHOT_DATASET[1]['answer'][:14].rstrip()} "
                f"Q: {FEW_SHOT_DATASET[2]['question'][:14].rstrip()} "
                "A:</s>"
            ),
        )
        self.assertEqual(
            self.tokenizer.decode(mapped_dataset[0]["labels"]),
            FEW_SHOT_DATASET[2]["answer"][:14].rstrip(),
        )

        # do it for the other few shot sample.
        self.assertEqual(
            self.tokenizer.decode(mapped_dataset[1]["input_ids"]),
            (
                f"Q: {FEW_SHOT_DATASET[3]['question'][:14].rstrip()} "
                f"A: {FEW_SHOT_DATASET[3]['answer'][:14].rstrip()} "
                f"Q: {FEW_SHOT_DATASET[4]['question'][:14].rstrip()} "
                f"A: {FEW_SHOT_DATASET[4]['answer'][:14].rstrip()} "
                f"Q: {FEW_SHOT_DATASET[5]['question'][:14].rstrip()} "
                "A:</s>"
            ),
        )

        self.assertEqual(
            self.tokenizer.decode(mapped_dataset[1]["labels"]),
            FEW_SHOT_DATASET[5]["answer"][:14].rstrip(),
        )
