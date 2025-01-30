import pandas as pd
from datasets import load_dataset
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import LLM
import json
import re
from tqdm import tqdm

def validate_dataset(data):
    
    for d in data:
        for k in ['subject', 'question', 'choices', 'answer']:
            if k not in d:
                print(f"Missing key: {k}")
                print(d)
                print("\n\n\n")
                
        for k in ['subject', 'question']:
            if not isinstance(d[k], str):
                print(f"Invalid type for key: {k}")
                print(d)
                print("\n\n\n")
                
        if not isinstance(d['choices'], list):
            print("Invalid type for key: choices")
            print(d)
            print("\n\n\n")
                
        if not isinstance(d['answer'], int):
            print("Invalid type for key: answer")
            print(d)
            print("\n\n\n")
            
    print("Data validation complete")

EVAL_PROMPT = """
The following are multiple-choice questions about {subject}. Please answer with the letter of the correct choice (A, B, C, D, ...) only. Answer the letter only. Do not need Explanation.

Output a valid json code blcok enclosed within a pair of triple quotes, with only 1 key being "answer" and the value being the letter of the correct choice.
For example: 
```json
{{
    "answer": "A"
}}
```
Question: {question}
Choices:
{choices}
"""

def format_mmlu_prompt(sample):
    return EVAL_PROMPT.format(
        subject=sample["subject"],
        question=sample["question"],
        choices="\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(sample["choices"])),
    )
    
def find_json_blocks(input_string):
    # Regular expression to find JSON code blocks
    json_blocks = re.findall(r'```json(.*?)```', input_string, re.DOTALL)
    
    # Parse each JSON block and store in a list
    json_dicts = []
    for block in json_blocks:
        try:
            json_dict = json.loads(block.strip())
            json_dicts.append(json_dict)
        except json.JSONDecodeError:
            print("Invalid JSON block found and skipped.")
    
    return json_dicts

class EvaluateMMLU:
    def __init__(self, model:LLM = None):
        if not model:
            self.llm = Ollama(model = 'qwen2.5:7b')
            
    def evalute_single(self, sample) -> dict:
        
        prompt = format_mmlu_prompt(sample)
        
        try:
            res = self.llm.complete(prompt)
            answer = find_json_blocks(res.text)[-1]['answer']
            ground_truth = chr(65 + sample['answer'])
            correct = answer == ground_truth
            return {
                "prompt": prompt, 
                "response": res.text, 
                "answer": answer, 
                "ground_truth": ground_truth, 
                "correct": correct}
        except Exception as e:
            print("Error in evaluating the sample")
            print(e)
            return {
                "prompt": prompt, 
                "response": "Error in evaluating the sample", 
                "answer": None, 
                "ground_truth": None, 
                "correct": False}
            
    def evaluate_dataset(self, data):
        results = []
        for sample in tqdm(data, desc="Evaluating:"):
            result = self.evalute_single(sample)
            results.append(result)
            
        json.dump(results, open("results.json", "w"), ensure_ascii=False, indent=4)
        return results
        
    
data = load_dataset("./data/mmlu","all", split = "test")    
evaluator = EvaluateMMLU()
evaluator.evaluate_dataset(data.select(range(100)))