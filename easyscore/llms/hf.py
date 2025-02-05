from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class HuggingfaceLLM:
    model_name_or_path: str
    
    def __post_init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def chat(self, messages:List[dict]):
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=4096
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    def complete(self, text:str):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text}
        ]
        return self.chat(messages)
    
    
    
