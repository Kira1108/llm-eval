from datasets import load_dataset

def load_mmlu(*args, **kwargs):
    return load_dataset(*args, **kwargs)

def load_mmlu_pro(*args, **kwargs):
    dataset = load_dataset(*args, **kwargs)
    dataset = dataset.rename_columns({
        "category":"subject",
        "answer":"answer_letter",
        "answer_index": 'answer',
        "options":"choices"
    })
    return dataset