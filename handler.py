from typing import Dict, List, Any

import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


class EndpointHandler:
    def __init__(self, path=""):
        # access_token = os.environ["HUGGINGFACE_TOKEN"]
        # login(token=access_token)
        # Load the model and tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=path,  # Use the current directory path
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)
        self.device = next(self.model.parameters()).device

        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="phi-3",
        )

    def __call__(self, data: Dict[str, Any]) -> List[str]:
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", {})

        tokenized_input = self.tokenizer.apply_chat_template(
            inputs,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            output = self.model.generate(
                input_ids=tokenized_input,
                **parameters
            )

        full_response = self.tokenizer.decode(output, skip_special_tokens=True)
        return full_response
