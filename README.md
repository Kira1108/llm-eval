# LLM Evaluation

**开源的大模型Benchmark框架**    
[Lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main)    
[Deep-eval](https://docs.confident-ai.com/docs/benchmarks-introduction)    
[Prompt-bench](https://github.com/microsoft/promptbench)

这些评估框架的问题在于，写死Prompt以及输出的解析方式，对于输出格式有变化的模型，比如长思考模型，适应性不是很好。

由于各种benchmark数据集的格式不统一，需要进行不同的处理，所以`Lm-evaluation-harness`对不同的任务写了不同的处理类，不同的prompt。
部分结果需要正则解析
```yaml
description: "The following are multiple choice questions (with answers) about chemistry. Think step by step and then finish your answer with \"the answer is (X)\" where X is the correct letter choice.\n"
include: "_default_template_yaml"
task: "mmlu_pro_chemistry"
task_alias: "chemistry"
process_docs: !function utils.process_chemistry
```

DeepEval同理，为每种Task定义了一个专门的Template
```python
from deepeval.benchmarks.mmlu.task import MMLUTask


class MMLUTemplate:

    # Most of this template was taken from MMLU Github Repo
    # The output confinement is a novel addition, since the original code
    # outputted log_probabilties for each answer choice

    @staticmethod
    def generate_output(
        input: str, train_set: object, task: MMLUTask, n_shots: int
    ):
        prompt = "The following are multiple choice questions (with answers) about{}.\n\n"
        prompt = prompt.format(MMLUTemplate.format_subject(task.value))
        for i in range(n_shots):
            prompt += MMLUTemplate.format_question(train_set[i])
        prompt += input
        return prompt

    @staticmethod
    def format_question(data: dict, include_answer: bool = True):
        prompt = data["input"]
        choices = ["A", "B", "C", "D"]
        for j in range(len(choices)):
            choice = choices[j]
            prompt += "\n{}. {}".format(choice, data[choice])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(data["target"])
        return prompt

    @staticmethod
    def format_subject(subject: str):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s
```

但是DeepEval文档中说，大多数需要输出json
**语言模型基准测试中的问题**
- 输出格式不正确：
许多公共基准测试要求特定的输出格式，通常是单个字母（例如A、B、C、D），因为这些基准测试通常是多项选择题（MCQ）的形式。
如果语言模型（LLM）未能生成严格符合这种格式的输出（例如，它生成了完整的句子、解释或其他不符合要求的文本），基准测试工具可能无法正确解释这些回答。
这种不匹配可能导致错误的评分，使得模型的表现看起来很差，即使它实际上理解了内容。
- 基准测试分数的影响：
当语言模型未能生成预期的单字母输出时，基准测试工具可能会将这些回答视为错误或无效。
这可能导致分数异常偏低，无法准确反映模型的实际能力。


**下载数据集**
- 安装Huggingface-CLI
- 配置国内镜像

```bash
pip install huggingface-hub
mkdir data
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download cais/mmlu --local-dir data/mmlu --repo-type dataset
```

**数据集**：
[MMLU](https://huggingface.co/datasets/cais/mmlu), [MMLU-PRO](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro?row=13)


格式：
- `question`: 问题, string
- `answer`:正确答案，整数，A-0, B-1, C-2, D-3, ....
- `choices`: 数组，选项， list[string]
- `subject`: 问题分类

**输出**

json code block
```json
{
    "answer": "A" # single letter
}
```


**使用**
```python
from easyscore.eval_tasks import MCQTask
from llama_index.llms.ollama import Ollama
from easyscore.data_loader import load_mmlu_pro

pro = load_mmlu_pro()
m = Ollama("qwen2.5")

small_dataset = pro.shuffle().select(range(10))
task = MCQTask(llm = m, data = small_dataset)
task.evaluate_dataset()
```


```python
from easyscore.eval_tasks import MCQTask,MathTask
from llama_index.llms.ollama import Ollama
from easyscore.data_loader import load_mmlu_pro,load_gsm8k

math_dataset = load_gsm8k().shuffle().select(range(10))

llm = Ollama("qwen2.5")
task = MathTask(llm = llm, data = math_dataset)
task.evaluate_dataset()
```