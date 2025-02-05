pip install huggingface-hub
mkdir -p data
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download cais/mmlu --local-dir data/mmlu --repo-type dataset
huggingface-cli download TIGER-Lab/MMLU-Pro --local-dir data/mmlu-pro --repo-type dataset