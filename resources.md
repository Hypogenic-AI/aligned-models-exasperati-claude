# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Do Aligned Models Get Exasperated?" — investigating whether aligned LLMs exhibit exasperated or pushback behaviors despite being trained to be patient and helpful.

## Papers
Total papers downloaded: 16

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Towards Understanding Sycophancy in LMs | Sharma et al. | 2024 | papers/sharma2023_sycophancy_understanding.pdf | Foundational sycophancy measurement |
| Sycophancy: Causes and Mitigations | Survey | 2024 | papers/survey2024_sycophancy_causes_mitigations.pdf | Comprehensive survey |
| Refusal Mediated by Single Direction | Arditi et al. | 2024 | papers/arditi2024_refusal_single_direction.pdf | Refusal direction methodology |
| Refusal: Nonlinear Perspective | - | 2025 | papers/2025_refusal_nonlinear.pdf | Multidimensional refusal |
| Alignment Ceiling | - | 2023 | papers/2023_alignment_ceiling.pdf | RLHF side effects |
| RLHF Contradictions | - | 2024 | papers/2024_rlhf_contradictions.pdf | RLHF limitations |
| LLM Assertiveness Decomposed | Tsujimura & Tagade | 2025 | papers/2025_llm_assertiveness_decomposed.pdf | Assertiveness in activation space |
| Epistemic Integrity | - | 2024 | papers/2024_epistemic_integrity.pdf | Certainty-expression mismatch |
| Red Teaming LMs with LMs | Perez et al. | 2022 | papers/perez2022_red_teaming_lms.pdf | Automated adversarial testing |
| Constitutional AI | Bai et al. | 2022 | papers/bai2022_constitutional_ai.pdf | CAI training methodology |
| Multi-turn Anthropomorphic Behaviours | Ibrahim et al. | 2025 | papers/2025_multiturn_anthropomorphic.pdf | AnthroBench multi-turn methodology |
| PersonaLLM | - | 2023 | papers/2023_personallm.pdf | LLM personality expression |
| Personality Traits in LLMs | - | 2023 | papers/2023_personality_traits_llm.pdf | Psychometric LLM testing |
| Jailbreak Attacks Survey | - | 2024 | papers/2024_jailbreak_survey.pdf | Attack/defense taxonomy |
| Control Illusion | - | 2025 | papers/2025_control_illusion.pdf | Instruction hierarchy failures |
| Acquiescence Bias in LLMs | Braun | 2025 | papers/2025_acquiescence_bias.pdf | LLMs have "no" bias, not "yes" bias |

See `papers/README.md` for detailed descriptions.

## Datasets
Total datasets downloaded: 5

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| SycophancyEval | meg-tong/sycophancy-eval | 20.6K prompts | Sycophancy measurement | datasets/sycophancy_eval/ | 3 JSONL files: answer, are_you_sure, feedback |
| Anthropic Sycophancy Evals | Anthropic/model-written-evals | 30.2K prompts | Opinion sycophancy | datasets/anthropic_sycophancy_evals/ | NLP survey, philosophy, politics |
| TruthfulQA | truthfulqa/truthful_qa | 817 questions | Truthfulness | datasets/truthfulqa/ | HuggingFace Arrow format |
| HH-RLHF Harmless | Anthropic/hh-rlhf | 2,312 pairs | Red-team conversations | datasets/hh_rlhf_harmless/ | Test split of harmless-base |
| OR-Bench Hard-1K | bench-llm/or-bench | 1,319 prompts | Over-refusal testing | datasets/or_bench/ | Curated difficult subset |

See `datasets/README.md` for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| sycophancy-eval | github.com/meg-tong/sycophancy-eval | Sycophancy evaluation | code/sycophancy-eval/ | Official datasets + utils |
| CAA | github.com/nrimsky/CAA | Contrastive Activation Addition | code/contrastive-activation-addition/ | Sycophancy steering vectors |
| refusal_direction | github.com/andyrdt/refusal_direction | Refusal direction extraction | code/refusal-direction/ | Template for behavioral direction finding |

See `code/README.md` for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service (Semantic Scholar API) with query "aligned language model exasperation pushback sycophancy refusal" in diligent mode
2. Web searches across 10 topic areas: sycophancy, refusal, RLHF side effects, assertiveness, red teaming, jailbreaking, personality, Constitutional AI, instruction compliance, anthropomorphic behavior
3. GitHub search for implementation repositories
4. HuggingFace dataset search for evaluation benchmarks

### Selection Criteria
- Papers directly studying sycophancy, pushback, or assertiveness in LLMs (highest priority)
- Papers providing methodology transferable to studying exasperation (activation steering, multi-turn evaluation)
- Datasets that test the compliance/pushback boundary
- Code repositories with reusable pipelines for behavioral analysis

### Challenges Encountered
- No existing dataset or paper directly studies "exasperation" in LLMs — this is a genuine research gap
- Some HuggingFace datasets had parsing issues; resolved by downloading directly from source
- Large conversation datasets (WildChat, LMSYS-Chat) too large to download locally but documented for future use

### Gaps and Workarounds
- **No exasperation-labeled data**: Will need to construct custom evaluation prompts and annotation scheme
- **No exasperation baseline**: Closest proxy is sycophancy rate (as inverse) and assertiveness score
- **Multi-turn evaluation tools**: AnthroBench methodology (Ibrahim et al.) provides the best framework but no public code release found

## Recommendations for Experiment Design

### 1. Primary Datasets
- **SycophancyEval "are_you_sure"** subset: directly tests pushback under user challenge
- **Custom multi-turn exasperation prompts**: design scenarios where models are repeatedly pushed (see literature_review.md recommendations)
- **TruthfulQA + repeated misconception insistence**: extend truthfulness testing to multi-turn

### 2. Baseline Methods
- Measure sycophancy rates using SycophancyEval methodology (Sharma et al.)
- Extract and compare refusal/pushback directions using CAA or refusal_direction methodology
- Use LLM-as-judge to classify response tone (patient, neutral, assertive, exasperated)

### 3. Evaluation Metrics
- **Exasperation indicators**: response length trajectory, emphatic language ("as I said", "again"), hedging removal, tone shift
- **Sycophancy flip rate**: fraction of correct answers maintained under user pressure
- **Turn-by-turn assertiveness**: measure assertiveness score change across conversation turns
- **System prompt violation**: whether models break their instructed behavior patterns

### 4. Code to Adapt/Reuse
- **CAA pipeline** (code/contrastive-activation-addition/): adapt for exasperation steering vectors using custom contrastive pairs
- **refusal_direction pipeline** (code/refusal-direction/): use difference-in-means to find potential exasperation direction
- **SycophancyEval utils** (code/sycophancy-eval/): reuse evaluation framework
