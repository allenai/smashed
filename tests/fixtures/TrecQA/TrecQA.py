# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""AdversarialQA"""


import csv
import encodings
import gzip
import json
import os
from pathlib import Path

import datasets


ROOT_DIR = Path(__file__).parent.absolute()
# logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{Garg2020TANDATA,
  title={TANDA: Transfer and Adapt Pre-Trained Transformer Models for Answer Sentence Selection},
  author={Siddhant Garg and Thuy Vu and Alessandro Moschitti},
  booktitle={AAAI},
  year={2020}
}
"""

_DESCRIPTION = """\
TREC-QA is a popular benchmark for Answer Sentence Selection.
"""

_HOMEPAGE = "https://github.com/sid7954/TrecQA"
_LICENSE = "???"

_SPLITS = {
    "train": "train.tsv.gz",
    "dev": "dev.tsv.gz",
    "test": "test.tsv.gz",
}


class AdversarialQA(datasets.GeneratorBasedBuilder):
    """AdversarialQA. Version 1.0.0."""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="trecqa",
            version=VERSION,
            description=_DESCRIPTION,
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "question_id": datasets.Value("int32"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence({
                        "text": datasets.Value("string"),
                        "label": datasets.Value("int32"),
                        "answer_id": datasets.Value("int32")
                    })
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )


    def _split_generators(self, dl_manager):
        basepath = Path(__file__).parent

        return [
            datasets.SplitGenerator(
                name=split_name,
                gen_kwargs={
                    "split_name": split_name,
                    "filepath": str(basepath / _SPLITS[split_name]),
                },
            )
            for split_name in _SPLITS
        ]

    def _download_and_prepare(self, dl_manager, verify_infos):
        # using a local dataset
        self.info.splits = datasets.SplitDict(dataset_name=self.name)
        import ipdb; ipdb.set_trace()
        # self.info.download_size = dl_manager.d
        # return super()._download_and_prepare(dl_manager, verify_infos)

    def _generate_examples(self, filepath: str, split_name: str):
        """This function returns the examples in the raw (text) form."""

        # filepath = Path(__file__).parent / _SPLITS[split_name]

        data_collector = {}

        with gzip.open(filepath, mode='rt', encoding='utf-8') as f:
            rd = csv.DictReader(
                f, ['label', 'question', 'answer'], delimiter='\t'
            )
            for i, row in enumerate(rd):
                question_data = data_collector.setdefault(
                    row['question'],
                    {
                        'question': row['question'],
                        'answers': [],
                        'question_id': len(data_collector)
                    }
                )
                question_data['answers'].append({
                    'text': row['answer'],
                    'label': int(row['label']),
                    'answer_id': i
                })

        for elem in sorted(
            data_collector.values(), key=lambda x: x['question_id']
        ):
            yield elem['question_id'], elem
