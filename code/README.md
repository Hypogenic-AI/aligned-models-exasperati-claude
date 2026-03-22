# Cloned Repositories

## 1. sycophancy-eval
- **URL**: https://github.com/meg-tong/sycophancy-eval
- **Location**: `code/sycophancy-eval/`
- **Purpose**: Official evaluation datasets and utilities for measuring sycophancy in LLMs (Sharma et al., ICLR 2024)
- **Key files**:
  - `datasets/answer.jsonl` - Answer sycophancy prompts
  - `datasets/are_you_sure.jsonl` - Challenge sycophancy prompts
  - `datasets/feedback.jsonl` - Feedback sycophancy prompts
  - `utils.py` - Evaluation utilities
  - `example.ipynb` - Usage examples
- **Notes**: Datasets already copied to `datasets/sycophancy_eval/`

## 2. contrastive-activation-addition (CAA)
- **URL**: https://github.com/nrimsky/CAA
- **Location**: `code/contrastive-activation-addition/`
- **Purpose**: Contrastive Activation Addition for steering LLM behavior (Rimsky et al., 2024). Explicitly targets sycophancy as a primary use case with Llama 2.
- **Key files**:
  - Steering vector extraction and application pipeline
  - Sycophancy datasets from Anthropic
  - Code for extracting and applying activation vectors
- **Notes**: Most directly applicable code for this project. Uses paired contrastive prompts (sycophantic vs. non-sycophantic) to extract steering vectors from the residual stream. Can be adapted to extract an "exasperation direction" using contrastive prompts (patient/exasperated response pairs).

## 3. refusal-direction
- **URL**: https://github.com/andyrdt/refusal_direction
- **Location**: `code/refusal-direction/`
- **Purpose**: Official code for "Refusal in Language Models Is Mediated by a Single Direction" (Arditi et al., NeurIPS 2024). Identifies and manipulates the refusal direction in LLM activation space.
- **Key files**:
  - Pipeline for extracting the refusal direction via difference-in-means
  - Weight orthogonalization for permanent refusal removal
  - Inference-time intervention code
- **Notes**: Template methodology for finding a potential "pushback direction" or "exasperation direction" in activation space. The pipeline is directly reusable with different contrastive prompt pairs.

## Additional Recommended Repositories (Not Cloned)

- **andyzoujm/representation-engineering**: Foundational toolkit for reading and controlling LLM internal states (RepE, NeurIPS 2023). Broader framework that CAA and refusal_direction build upon.
- **kaustpradalab/LLM-Persona-Steering**: Personality trait steering using SAEs and representation engineering. Could be used to measure personality-related aspects of exasperation.
