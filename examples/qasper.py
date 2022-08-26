# type: ignore

import datasets
from itertools import chain
from transformers.models.auto.tokenization_auto import AutoTokenizer
from smashed.interfaces.huggingface import (
    TokenizerMapper,
    MultiSequenceStriderMapper,
    TokensSequencesPaddingMapper,
    AttentionMaskSequencePaddingMapper,
    SequencesConcatenateMapper,
    Python2TorchMapper,
    ChangeFieldsMapper
)

# create tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-uncased',
)

# Create dataset
qasper = datasets.load_dataset('qasper', split='train')
qasper_sentences = qasper.map(
    lambda row: {
        'sentences': list(chain.from_iterable(row['full_text']['paragraphs']))
    },
    remove_columns=qasper.column_names
).filter(lambda row: len(row['sentences']) > 0)

# build pipeline
pipeline = TokenizerMapper(
    input_field='sentences',
    tokenizer=tokenizer,
    add_special_tokens=False,
    truncation=True,
    max_length=80
) >> ChangeFieldsMapper(
    drop_fields=['sentences']
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
) >> SequencesConcatenateMapper(
) >> Python2TorchMapper()


# process the dataset all in one go
qasper_processed = pipeline.map(qasper_sentences, num_proc=4)
print(qasper_processed[0])
