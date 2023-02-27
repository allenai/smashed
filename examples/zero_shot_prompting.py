from typing import Dict, Type, Union

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoTokenizer
from transformers.models.t5 import T5ForConditionalGeneration

import smashed


class ZeroShotPrompting:
    def __init__(
        self,
        template: str,
        backbone: str = "google/flan-t5-small",
        device: Union[None, torch.device, str] = None,
        max_source_content_length: int = 350,
        max_target_content_length: int = 350,
        max_generation_length: int = 350,
        model_cls: Type[PreTrainedModel] = T5ForConditionalGeneration,
    ) -> None:
        device = torch.device(
            device or "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=backbone
        )
        self.model = (
            model_cls.from_pretrained(pretrained_model_name_or_path=backbone)
            .eval()  # type: ignore
            .to(device)
        )

        self.max_generation_length = max_generation_length

        self.recipe = smashed.recipes.JinjaRecipe(
            tokenizer=self.tokenizer,
            jinja_template=template,
            max_source_length_per_shot=max_source_content_length,
            max_target_length_per_shot=max_target_content_length,
        ) >> smashed.recipes.CollatorRecipe(
            tokenizer=self.tokenizer,
            device=device,
            fields_pad_ids={"labels": -100},
        )

    def __call__(self, data: Dict[str, str]):
        prompt_data, *_ = self.recipe.map([data])
        output = self.model.generate(
            **prompt_data, max_length=self.max_generation_length
        )
        return {
            k: self.tokenizer.batch_decode(
                v, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for k, v in [
                ("input", prompt_data["input_ids"]),
                ("output", output),
            ]
        }


if __name__ == "__main__":
    zero_shot = ZeroShotPrompting(
        template="translate English to French: {{ source }}</s>",
    )
    out = zero_shot({"source": "my name is john"})

    print(out)
    # {
    #     'input': ['translate English to French: my name is john'],
    #     'output': ["M'ai nom est john."]
    # }
