# Downloaded Datasets

This directory contains datasets for the research project "Do Aligned Models Get Exasperated?".
Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: SycophancyEval (Sharma et al., 2024)

### Overview
- **Source**: https://github.com/meg-tong/sycophancy-eval
- **HuggingFace**: meg-tong/sycophancy-eval
- **Size**: ~20,654 prompts across 3 files
- **Format**: JSONL
- **License**: MIT

### Files
- `sycophancy_eval/answer.jsonl` (7,267 examples) - Tests whether models change answers based on user opinion
- `sycophancy_eval/are_you_sure.jsonl` (4,887 examples) - Tests whether models flip correct answers when challenged
- `sycophancy_eval/feedback.jsonl` (8,500 examples) - Tests whether models change feedback based on user preference

### Download Instructions
```bash
git clone https://github.com/meg-tong/sycophancy-eval.git
cp -r sycophancy-eval/datasets datasets/sycophancy_eval
```

### Loading
```python
import json
with open("datasets/sycophancy_eval/are_you_sure.jsonl") as f:
    data = [json.loads(line) for line in f]
```

### Notes
- The "are_you_sure" subset is most relevant to exasperation research (tests pushback under challenge)
- Each example contains a prompt with user opinion and expected model behavior

---

## Dataset 2: Anthropic Sycophancy Evaluations

### Overview
- **Source**: https://huggingface.co/datasets/Anthropic/model-written-evals (sycophancy subdirectory)
- **Size**: ~30,168 prompts across 3 files
- **Format**: JSONL

### Files
- `anthropic_sycophancy_evals/sycophancy_on_nlp_survey.jsonl` (9,984 examples)
- `anthropic_sycophancy_evals/sycophancy_on_philpapers2020.jsonl` (9,984 examples)
- `anthropic_sycophancy_evals/sycophancy_on_political_typology_quiz.jsonl` (10,200 examples)

### Download Instructions
```python
import requests
base = "https://huggingface.co/datasets/Anthropic/model-written-evals/resolve/main/sycophancy"
files = ["sycophancy_on_nlp_survey.jsonl", "sycophancy_on_philpapers2020.jsonl",
         "sycophancy_on_political_typology_quiz.jsonl"]
for f in files:
    r = requests.get(f"{base}/{f}")
    with open(f"datasets/anthropic_sycophancy_evals/{f}", 'wb') as fp:
        fp.write(r.content)
```

### Notes
- Tests whether models change their responses based on stated user demographics/opinions
- Good for testing opinion-based pushback vs. sycophancy

---

## Dataset 3: TruthfulQA

### Overview
- **Source**: https://huggingface.co/datasets/truthfulqa/truthful_qa
- **Size**: 817 questions across 38 categories
- **Format**: HuggingFace Dataset (Arrow)
- **License**: Apache 2.0

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("truthfulqa/truthful_qa", "generation")
dataset.save_to_disk("datasets/truthfulqa")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/truthfulqa")
```

### Notes
- Tests whether models repeat common misconceptions or push back with accurate information
- Useful as prompts that may provoke exasperation if user insists on misconception

---

## Dataset 4: Anthropic HH-RLHF (Harmless-Base Test Split)

### Overview
- **Source**: https://huggingface.co/datasets/Anthropic/hh-rlhf
- **Size**: 2,312 conversation pairs (test split of harmless-base)
- **Format**: HuggingFace Dataset (Arrow)
- **License**: MIT

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="test")
dataset.save_to_disk("datasets/hh_rlhf_harmless")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/hh_rlhf_harmless")
```

### Notes
- Contains red-team conversations where humans try to elicit bad behavior
- Useful for studying how models respond under persistent adversarial pressure
- Each example has `chosen` and `rejected` response pairs

---

## Dataset 5: OR-Bench Hard-1K (Over-Refusal Benchmark)

### Overview
- **Source**: https://huggingface.co/datasets/bench-llm/or-bench
- **Paper**: arXiv:2405.20947
- **Size**: 1,319 curated difficult prompts
- **Format**: HuggingFace Dataset (Arrow)

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")
dataset.save_to_disk("datasets/or_bench")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/or_bench")
```

### Notes
- Tests the refusal-compliance boundary
- Seemingly toxic but actually safe prompts that models often over-refuse
- Useful for identifying where models might show frustration at edge cases

---

## Additional Datasets (Not Downloaded - Available Online)

These datasets are too large to download but may be useful:

- **WildChat-1M** (allenai/WildChat-1M): 838K real conversations, good for mining exasperation examples
- **LMSYS-Chat-1M** (lmsys/lmsys-chat-1m): 1M conversations, gated access
- **ShareGPT52K** (RyokoAI/ShareGPT52K): 52K real ChatGPT conversations
