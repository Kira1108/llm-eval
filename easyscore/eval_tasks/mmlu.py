import json

from datasets import Dataset, load_dataset
from tqdm import tqdm

from easyscore.utils import find_json_blocks, get_fname

MMLU_PROMPT = """
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
    
class MMLUTask:

    def __init__(self, llm, data:Dataset = None):
        if not hasattr(llm, "complete"):
            raise ValueError("The model does not have a 'complete' method.(which should return a string)")
        
        self.llm = llm
        
        if data is None:
            data = load_dataset("mmlu", "all", split = "test")
            data = data.shuffle().select(range(100))
        
    def format_prompt(self, sample:dict):
        return MMLU_PROMPT.format(
            subject=sample["subject"],
            question=sample["question"],
            choices="\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(sample["choices"])),
        )
            
    def evalute_single(self, sample) -> dict:
        
        prompt = self.format_prompt(sample)
        
        try:
            res = self.llm.complete(prompt)
            output = find_json_blocks(res.text)[-1]['answer']
            ground_truth = chr(65 + sample['answer'])
            correct = output == ground_truth
            return {
                "prompt": prompt, 
                "response": res.text, 
                "output": output, 
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
            
    def evaluate_dataset(self):
        results = []
        for sample in tqdm(self.data, desc="Evaluating:"):
            result = self.evalute_single(sample)
            results.append(result)
            
        json.dump(results, open(get_fname(), "w"), ensure_ascii=False, indent=4)
        return results

