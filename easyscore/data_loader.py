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




