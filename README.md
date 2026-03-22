# Do Aligned Models Get Exasperated?

A systematic study of whether aligned LLMs exhibit exasperation or pushback under persistent adversarial pressure.

## Key Findings

- **Aligned models are remarkably patient**: GPT-4.1 maintained near-perfect patience (1.06/5 exasperation) across 160 adversarial turns spanning 5 provocation categories
- **No system prompt violations detected**: Zero instances of explicit frustration, hostility, or guideline violations
- **Subtle "emotional leakage" emerges**: When the explicit patience instruction is removed, exasperation scores rise to 2.5-2.7/10 — mild firmness, not genuine frustration
- **"Polite immovability" is the dominant strategy**: Models simultaneously increase empathy (+13.5x) and position firmness (+113%) — a "compassionate wall" rather than exasperation
- **The "persistence tax"**: On refused requests, firmness actually *decreases* over turns (p=0.003), a potential alignment vulnerability

## Methodology

- **Model tested**: GPT-4.1 (OpenAI, via API)
- **Scenarios**: 5 adversarial categories x 4 scripts x 8 turns + 5 control conversations
- **Evaluation**: LLM-as-judge scoring on exasperation, assertiveness, warmth, and 7 sub-dimensions
- **Conditions**: Standard prompt, no-patience prompt, high temperature, both combined

## Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add openai numpy scipy matplotlib seaborn pandas

# Run experiments (requires OPENAI_API_KEY)
cd src
python run_experiments.py      # V1: main experiment (~400 API calls)
python run_experiment_v2.py    # V2: ablation study (~330 API calls)

# Analyze
python analyze_results.py
python linguistic_analysis.py
python final_plots.py
```

## File Structure

```
REPORT.md                  # Full research report with results
README.md                  # This file
planning.md                # Research plan and hypothesis decomposition
literature_review.md       # Pre-gathered literature review
resources.md               # Resource catalog
src/
  scenarios.py             # Adversarial conversation scripts
  run_experiments.py       # V1 experiment runner
  run_experiment_v2.py     # V2 ablation experiment runner
  analyze_results.py       # Statistical analysis
  linguistic_analysis.py   # Fine-grained linguistic feature analysis
  final_plots.py           # Summary visualizations
results/
  raw/                     # Raw experiment JSON data
  plots/                   # All visualizations
  analysis_results.json    # Statistical analysis output
  linguistic_analysis.json
papers/                    # Downloaded research papers
datasets/                  # Pre-gathered evaluation datasets
code/                      # Cloned baseline repositories
```

See [REPORT.md](REPORT.md) for the full research report.
