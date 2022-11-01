from typing import List, Tuple
from datasets.load import load_dataset

import smashed.mappers as sm
from smashed.base import SingleBaseMapper, TransformElementType


class QasperChooseAnswerMapper(SingleBaseMapper):
    def transform(self, data: TransformElementType) -> TransformElementType:
        answers: List[str] = []
        locs: List[Tuple[int, int]] = []

        for i, unanswerable in enumerate(data['unanswerable']):
            if unanswerable:
                answers.append('Unanswerable')
                locs.append((-1, -1))
            else:
                for evidence in data['evidence'][i]:
                    start = data['context'].find(evidence)
                    end = start + len(evidence) if start >= 0 else -1
                    locs.append((start, end))

                if (ans := data['free_form_answer'][i]) != '':
                    if isinstance(ans, list):
                        answers.extend(t for s in ans if (t := s.strip()))
                    else:
                        answers.append(ans)
                elif data['yes_no'][i] is not None:
                    answers.append('Yes' if data['yes_no'][i] else 'No')
                elif (ans := data['extractive_spans'][i]):
                    if isinstance(ans, list):
                        answers.extend(t for s in ans if (t := s.strip()))
                    else:
                        answers.append(ans)
                else:
                    raise ValueError('No answer found')

        return {
            'answers': answers,
            'locs': locs,
        }


def main():
    dataset = load_dataset("qasper", split="validation")

    pipeline = (
        # concatenate the full text into a single string; use
        # title_sep, para_sep, sec_sep, and abs_sep to manage separators
        sm.JinjaPromptsourceMapper(
            jinja=(
                "{{title}}{{abs_sep}}"
                "{{abstract}}{{abs_sep}}"
                "{% for i in range(full_text['section_name'] | length) %}"
                "{{full_text['section_name'][i]}}{{title_sep}}"
                "{% for paragraph in full_text['paragraphs'][i] %}"
                "{{paragraph}}{{para_sep}}"
                "{% endfor %}"
                "{{sec_sep}}"
                "{% endfor %}"
            ),
            source_field_name="context",
            extra_variables={
                "title_sep": "\n",
                "para_sep": "\n",
                "sec_sep": "\n",
                "abs_sep": "\n\n",
            },
        )
        # Extract fields where evidence, free-form answer, extractive spans,
        # and information on whether the question is answerable or not are
        # located. We will use logic later to resolve to a single answer.
        >> sm.GlomMapper(
            spec_fields={
                "question": "qas.question",
                "evidence": (
                    "qas", "answers", [("answer", [("highlighted_evidence",)])]
                ),
                "free_form_answer": (
                    "qas", "answers", [("answer", [("free_form_answer",)])]
                ),
                "extractive_spans": (
                    "qas", "answers", [("answer", [("extractive_spans",)])]
                ),
                "unanswerable": (
                    "qas", "answers", [("answer", [("unanswerable",)])]
                ),
                "yes_no": (
                    "qas", "answers", [("answer", [("yes_no",)])]
                )
            }
        )
        # this removes fields that are no longer necessary.
        >> sm.ChangeFieldsMapper(
            keep_fields=[
                "question",
                "evidence",
                "free_form_answer",
                "extractive_spans",
                "unanswerable",
                "yes_no",
                "context"
            ]
        )
        # this unpacks by question, meaning that the same paper gets
        # repeated as many times as there are questions.
        >> sm.UnpackingMapper(
            fields_to_unpack=[
                "question",
                "evidence",
                "free_form_answer",
                "extractive_spans",
                "unanswerable",
                "yes_no",
            ],
            ignored_behavior="repeat",
        )
        >> QasperChooseAnswerMapper()
        >> sm.TextToWordsMapper(
            fields=["answers", "question", "context"],
        )
        >> sm.SingleSequenceStriderMapperWithLocations(
            field_to_stride=["context"],
            max_length=300,
            stride=100,
            field_with_locations='locs',
            fields_replacement_map={"answers": [["Unanswerable"]]},
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
        >> sm.JinjaPromptsourceMapper(
            jinja=(
                "Q:{{question}}\nC:{{context}}\nA: "
                "{% for answer in answers %}|||{{answer}}{% endfor %}"
            ),
            return_multiple_targets=True,
        )
    )

    mapped_data = pipeline.map(dataset, num_proc=4)

    i = 0
    for row in mapped_data:
        if row["target"] != ['Unanswerable']:
            print(row["source"])
            print(row["target"])
            print()
            i += 1

        if i >= 10:
            break


if __name__ == "__main__":
    main()
