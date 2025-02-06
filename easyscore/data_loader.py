from datasets import load_dataset

def load_mmlu():
    return load_dataset("cais/mmlu", "all",split = 'test')

def load_mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro","default",split = 'test')
    
    # conform to the mmlu dataset format
    dataset = dataset.rename_columns({
        "category":"subject",
        "answer":"answer_letter",
        "answer_index": 'answer',
        "options":"choices"
    })
    return dataset


def split_answer(example):
    reasoning, answer = example['answer'].split("####")
    example['reasoning'] = reasoning.strip()
    example['answer'] = float(answer.strip().replace(",",""))
    example['subject'] = 'math'
    return example

def load_gsm8k():
    dataset = load_dataset("gsm8k",'main', split = 'test')
    return dataset.map(split_answer)




