![Colorful logo of smashed. It is the word smashed written in a playful font that vaguely looks like pipes.](https://github.com/allenai/smashed/raw/main/resources/smashed.png)

**S**equential **MA**ppers for **S**equences of **HE**terogeneous **D**ictionaries is a set of Python interfaces designed to apply transformations to samples in datasets, which are often implemented as sequences of dictionaries. To start, run

```bash
pip install smashed
```

## Example of Usage

Mappers are initialized and then applied sequentially. In the following example, we create a mapper that is applied to a samples, each containing a sequence of strings.
The mappers are responsible for the following operations.

1. Tokenize each sequence, cropping it to a maximum length if necessary.
2. Stride sequences together to a maximum length or number of samples.
3. Add padding symbols to sequences and attention masks.
4. Concatenate all sequences from a stride into a single sequence.

```python
import transformers
from smashed.mappers import (
    TokenizerMapper,
    MultiSequenceStriderMapper,
    TokensSequencesPaddingMapper,
    AttentionMaskSequencePaddingMapper,
    SequencesConcatenateMapper,
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-uncased',
)

mappers = [
    TokenizerMapper(
        input_field='sentences',
        tokenizer=tokenizer,
        add_special_tokens=False,
        truncation=True,
        max_length=80
    ),
    MultiSequenceStriderMapper(
        max_stride_count=2,
        max_length=512,
        tokenizer=tokenizer,
        length_reference_field='input_ids'
    ),
    TokensSequencesPaddingMapper(
        tokenizer=tokenizer,
        input_field='input_ids'
    ),
    AttentionMaskSequencePaddingMapper(
        tokenizer=tokenizer,
        input_field='attention_mask'
    ),
    SequencesConcatenateMapper()
]

dataset = [
    {
        'sentences': [
            'This is a sentence.',
            'This is another sentence.',
            'Together, they make a paragraph.',
        ]
    },
    {
        'sentences': [
            'This sentence belongs to another sample',
            'Overall, the dataset is made of multiple samples.',
            'Each sample is made of multiple sentences.',
            'Samples might have a different number of sentences.',
            'And that is the story!',
        ]
    }
]

for mapper in mappers:
    dataset = mapper.map(dataset)

print(len(dataset))

# >>> 5

print(dataset[0])

# >>> {
#    'input_ids': [
#        101,
#        2023,
#        2003,
#        1037,
#        6251,
#        1012,
#        102,
#        2023,
#        2003,
#        2178,
#        6251,
#        1012,
#        102
#    ],
#    'attention_mask': [
#        1,
#        1,
#        1,
#        1,
#        1,
#        1,
#        1,
#        1,
#        1,
#        1,
#        1,
#        1,
#        1
#    ]
# }
```

## Building a Pipeline

Mappers can also be composed into a pipeline using the `>>` (or `<<`) operator. For example, the code above can be rewritten as follows:

```python
pipeline = TokenizerMapper(
    input_field='sentences',
    tokenizer=tokenizer,
    add_special_tokens=False,
    truncation=True,
    max_length=80
) >> MultiSequenceStriderMapper(
    max_stride_count=2,
    max_length=512,
    tokenizer=tokenizer,
    length_reference_field='input_ids'
) >> TokensSequencesPaddingMapper(
    tokenizer=tokenizer,
    input_field='input_ids'
) >> AttentionMaskSequencePaddingMapper(
    tokenizer=tokenizer,
    input_field='attention_mask'
) >> SequencesConcatenateMapper()

dataset = ...

# apply the full pipeline to the dataset
pipeline.map(dataset)
```

## Dataset Interfaces Available

The initial version of SMASHED supports two interfaces for dataset:

1. **`interfaces.simple.Dataset`**: A simple dataset representation that is just a list of python dictionaries with some extra convenience methods to make it work with SMASHED. You can crate a simple dataset by passing a list of dictionaries to `interfaces.simple.Dataset`.
2. **HuggingFace `datasets` library**. SMASHED mappers work with any datasets from HuggingFace, whether it is a regular or iterable dataset.

## Developing SMASHED

To contribute to SMASHED, make sure to:

1. (If you are not part of AI2) Fork the repository on GitHub.
2. Clone it locally.
3. Create a new branch in for the new feature.
4. Install development dependencies with `pip install -r dev-requirements.txt`.
5. Add your new mapper or feature.
6. Add unit tests.
7. Run tests, linting, and type checking from the root directory of the repo:
    1. *Style:* `black .` (Should format for you)
    2. *Style:* `flake8 .`  (Should return no error)
    3. *Style:* `isort .` (Should sort imports for you)
    4. *Static type check:* `mypy .` (Should return no error)
    5. *Tests:* `pytest -v --color=yes tests/` (Should return no error)
8. Commit, push, and create a pull request.
9. Tag `soldni` to review the PR.

### A note about versioning

SMASHED follows [Semantic Versioning](https://semver.org/). In short, this means that the version number is MAJOR.MINOR.PATCH, where:

- MAJOR version when you make incompatible API changes,
- MINOR version when you add functionality in a backwards compatible manner; adding a mapper typically falls under this category, and
- PATCH version when you make backwards compatible bug fixes.
