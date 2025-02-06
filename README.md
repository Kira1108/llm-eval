# LLM Evaluation

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