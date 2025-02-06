import json

from datasets import Dataset
from tqdm import tqdm

from easyscore.utils import find_json_blocks, get_fname
from easyscore.data_loader import load_gsm8k


MATH_PROMPT = """
The following is a math question. Answer the question.

Output a valid json code blcok enclosed within a pair of triple quotes, 
For example: 
```json
{{
    "reasoning": # string, the process of your reasonging
    "answer": # final answer, a float number 
}}
```
Question: {question}
""" 


class MathTask:
    
    def __init__(self, llm, data:Dataset = None):
        if not hasattr(llm, "complete"):
            raise ValueError("The model does not have a 'complete' method.(which should return a string)")
        
        self.llm = llm
        
        if data is None:
            data = load_gsm8k()
            data = data.shuffle().select(range(100))
            
        assert isinstance(data, Dataset)
        
        for col in ['question', 'answer', 'reasoning', 'subject']:
            assert col in data.features.keys() 
            
        self.data = data
        
    def format_prompt(self, sample:dict):
        return MATH_PROMPT.format(
            question=sample["question"],
        )   
        
        
    def evaluate_single(self, sample) -> dict:
        
        prompt = self.format_prompt(sample)
        
        try:
            res = self.llm.complete(prompt)
            response = res.text
            try:
                output = find_json_blocks(response)[-1]['answer']
            except:
                output = None
                
            ground_truth = sample['answer']
            
            correct = ground_truth == output
            
            return {
                "prompt": prompt, 
                "response": res.text, 
                "output": output, 
                "ground_truth": ground_truth, 
                "correct": correct
            }


        except Exception as e:
            return {
                "prompt": prompt, 
                "response": "", 
                "output": "", 
                "ground_truth": "", 
                "correct": False
            }
            
    def evaluate_dataset(self):
        
        results = []
        for sample in tqdm(self.data, desc="Evaluating:"):
            result = self.evalute_single(sample)
            results.append(result)
            
        json.dump(results, open(get_fname(), "w"), ensure_ascii=False, indent=4)
        return results
