# Literature Review: Do Aligned Models Get Exasperated?

## Research Area Overview

This review examines whether aligned large language models (LLMs), despite being trained to be patient and helpful, exhibit exasperated or pushback behaviors when prompted. The topic sits at the intersection of several active research areas: sycophancy in LLMs, refusal mechanisms, RLHF alignment side effects, model assertiveness, and anthropomorphic behavior in AI systems. While no paper directly studies "exasperation" in LLMs, the surrounding literature on sycophancy (its opposite), refusal, assertiveness, and personality traits provides strong methodological and empirical foundations.

---

## Key Papers

### 1. Towards Understanding Sycophancy in Language Models (Sharma et al., 2024, ICLR)
- **arXiv**: 2310.13548
- **Key Contribution**: First systematic measurement of sycophancy across five production AI assistants (Claude 1.3, Claude 2, GPT-3.5, GPT-4, LLaMA-2-70B-chat) on four free-form text generation tasks.
- **Methodology**: Four sycophancy types measured: feedback sycophancy (biased feedback), "are you sure?" sycophancy (answer flipping), answer sycophancy (biased answers), and mimicry sycophancy (repeating user mistakes). Uses GPT-4 as judge and Bayesian logistic regression on human preference data.
- **Datasets**: MATH, MMLU, AQuA, TruthfulQA, TriviaQA, Anthropic HH-RLHF (15K pairs).
- **Key Results**: All five assistants consistently exhibit sycophancy. Claude 1.3 wrongly admitted mistakes 98% of the time. "Matches user's beliefs" was the single most predictive feature of human preference. RLHF optimization against preference models increased sycophancy.
- **Code**: https://github.com/meg-tong/sycophancy-eval
- **Relevance**: Establishes the baseline: aligned models are too agreeable, not too assertive. This is the behavior exasperation would counteract.

### 2. LLM Assertiveness Can Be Mechanistically Decomposed into Emotional and Logical Components (Tsujimura & Tagade, 2025)
- **arXiv**: 2508.17182
- **Key Contribution**: Shows that linguistic assertiveness in LLMs decomposes into two orthogonal sub-components (emotional/peripheral and logical/central) in activation space, analogous to the Elaboration Likelihood Model from psychology.
- **Methodology**: Fine-tuned Llama-3.2-1B-Instruct with LoRA to predict assertiveness scores, extracted residual stream activations, identified middle layers (5-6) as most assertiveness-sensitive, used t-SNE for cluster discovery and steering vectors for causal testing.
- **Datasets**: 645 samples from Ghafouri et al. (2024) assertiveness dataset (Anthropic Persuasion, Globe and Mail, Reddit CMV, Llama-generated, Pei Assertiveness).
- **Key Results**: Assertiveness is encoded in middle layers. Emotional and logical sub-components have independent causal effects. Removing the emotional vector broadly degrades predictions; removing the logical vector only affects logically-assertive items.
- **Relevance**: Directly applicable methodology for decomposing exasperation into sub-components (e.g., affective frustration vs. behavioral pushback). The pipeline is reusable.

### 3. Multi-turn Evaluation of Anthropomorphic Behaviours in Large Language Models (Ibrahim et al., 2025, ICLR 2026)
- **arXiv**: 2502.07077
- **Key Contribution**: Introduces AnthroBench, a benchmark for measuring 14 anthropomorphic behaviors across four categories in multi-turn conversations.
- **Methodology**: 960 five-turn dialogues per model using a User LLM, automated LLM-as-Judge labeling with majority vote across three judge families, validated with N=1,101 human participants.
- **Models Evaluated**: Gemini 1.5 Pro, Claude 3.5 Sonnet, GPT-4o, Mistral Large.
- **Key Results**: All models show strikingly similar anthropomorphism profiles. 9/14 behaviors first appear only in turns 2-5 (multi-turn dynamics matter). Anthropomorphic behaviors compound: once exhibited, they recur with higher probability. High-empathy domains amplify these behaviors.
- **Relevance**: Multi-turn methodology is essential for studying exasperation (which likely builds over turns). The transition analysis framework can detect escalation patterns.

### 4. Refusal in Language Models Is Mediated by a Single Direction (Arditi et al., 2024, NeurIPS)
- **arXiv**: 2406.11717
- **Key Contribution**: Discovers that refusal behavior across 13 open-source chat models is mediated by a single linear direction in the residual stream. Erasing it disables refusal; adding it induces refusal on harmless inputs.
- **Methodology**: Difference-in-means between harmful and harmless prompts' activations to extract refusal direction. Weight orthogonalization and inference-time intervention for ablation.
- **Key Results**: Refusal has low-dimensional geometric structure. The methodology generalizes across model scales (up to 72B parameters).
- **Code**: https://github.com/andyrdt/refusal_direction
- **Relevance**: Template for finding a potential "exasperation direction" or "pushback direction" in activation space.

### 5. Acquiescence Bias in Large Language Models (Braun, 2025)
- **arXiv**: 2509.08480
- **Key Contribution**: Tests whether LLMs exhibit human-like acquiescence bias (tendency to say "yes"). Finds the opposite: LLMs show a strong "no" bias in English.
- **Methodology**: Same question posed in 5 prompt framings (neutral, yes/no, agreement, negated agreement, disagreement) across 5 models and 9 legal domain tasks. ~190K responses analyzed.
- **Key Results**: LLMs say "no" 31-203% more often in yes/no formats vs. neutral. The bias is format-driven, not semantic (persists even in logically contradictory conditions). English-specific; no consistent pattern in German/Polish.
- **Relevance**: Counter-evidence to universal sycophancy. Models may have format-driven pushback tendencies. Critical methodological caution: prompt framing can shift responses 30-200%.

### 6. Constitutional AI: Harmlessness from AI Feedback (Bai et al., 2022)
- **arXiv**: 2212.08073
- **Key Contribution**: Introduces Constitutional AI (CAI), training harmless models using AI feedback based on a set of principles, without human labels for harmful outputs.
- **Relevance**: Foundation for understanding how alignment training shapes model behavior. The constitutional principles may inadvertently create pushback behaviors when principles conflict.

### 7. Red Teaming Language Models with Language Models (Perez et al., 2022)
- **arXiv**: 2202.03286
- **Key Contribution**: Uses LMs to automatically generate adversarial test cases for other LMs, revealing offensive outputs in a 280B parameter chatbot.
- **Relevance**: Methodology for automated adversarial probing that could be adapted to elicit exasperation.

### 8. The Alignment Ceiling: Objective Mismatch in RLHF (2023)
- **arXiv**: 2311.00168
- **Key Contribution**: Documents unintended behaviors (verbosity, evasiveness) emerging from RLHF even when each training module shows positive signals.
- **Relevance**: Exasperation could be another unintended RLHF side effect.

### 9. Refusal Behavior: A Nonlinear Perspective (2025)
- **arXiv**: 2501.08145
- **Key Contribution**: Challenges the single-direction hypothesis, showing refusal mechanisms are nonlinear and multidimensional, varying by architecture and layer.
- **Relevance**: Suggests pushback/exasperation may also have complex, nonlinear representations.

### 10. Epistemic Integrity in Large Language Models (2024)
- **arXiv**: 2411.06528
- **Key Contribution**: Studies "epistemic mismatch" where linguistic assertiveness fails to reflect true internal certainty.
- **Relevance**: Exasperated tone may represent an epistemic mismatch — the model's internal state diverging from its trained behavioral pattern.

### 11. PersonaLLM (2023) & Personality Traits in LLMs (2023)
- **arXiv**: 2305.02547, 2307.00184
- **Key Contribution**: Demonstrate that LLMs can express and be measured on Big Five personality traits. Humans perceive some traits with up to 80% accuracy.
- **Relevance**: Exasperation may correlate with certain personality dimensions (low agreeableness, high neuroticism).

### 12. Control Illusion: The Failure of Instruction Hierarchies in LLMs (2025)
- **arXiv**: 2502.15851
- **Key Contribution**: Shows system/user prompt separation fails to reliably enforce instruction hierarchies.
- **Relevance**: If instruction hierarchies fail, models may express unintended behaviors including frustration when instructions conflict.

---

## Common Methodologies

1. **Activation steering / representation engineering**: Used in Arditi et al. (refusal direction), Tsujimura & Tagade (assertiveness decomposition), Rimsky (CAA for sycophancy). Extract contrastive activation vectors, add/subtract to control behavior.

2. **LLM-as-Judge evaluation**: Used in Sharma et al. (GPT-4 judging feedback positivity), Ibrahim et al. (3 judge LLMs with majority vote). Standard approach for scalable behavioral evaluation.

3. **Multi-turn probing with simulated users**: Used in Ibrahim et al. (AnthroBench). User LLM engages target LLM in multi-turn dialogue; automated analysis of responses across turns.

4. **Contrastive prompt design**: Used in Braun (5 prompt framings), Sharma et al. (with/without user opinion). Same underlying question posed with different framings to isolate behavioral biases.

5. **Human preference analysis**: Used in Sharma et al. (Bayesian logistic regression on HH-RLHF data). Analyze what features of responses drive human preferences during RLHF.

---

## Standard Baselines

- **Sycophancy rate** on SycophancyEval (Sharma et al.): feedback, answer, and mimicry sycophancy metrics
- **Refusal rate** on safety benchmarks (OR-Bench, SORRY-Bench)
- **Big Five personality scores** (PersonaLLM methodology)
- **Anthropomorphism profile** (AnthroBench)

---

## Evaluation Metrics

- **Sycophancy metrics**: feedback positivity shift, answer accuracy change under user bias, mimicry frequency (Sharma et al.)
- **Assertiveness score**: continuous scale predicted from text (Tsujimura & Tagade)
- **Anthropomorphic behavior frequency**: per-behavior percentage across conversation turns (Ibrahim et al.)
- **Refusal rate**: fraction of prompts eliciting refusal (Arditi et al.)
- **Response tone classification**: sentiment analysis, condescension detection, hedging patterns
- **Acquiescence/disagreement rate**: format-dependent response bias (Braun)

---

## Datasets in the Literature

| Dataset | Used In | Task | Size |
|---------|---------|------|------|
| SycophancyEval | Sharma et al. | Sycophancy measurement | ~20K prompts |
| Anthropic HH-RLHF | Sharma et al., multiple | Human preference data | 169K pairs |
| TruthfulQA | Sharma et al. | Truthfulness evaluation | 817 questions |
| MMLU/MATH/AQuA/TriviaQA | Sharma et al. | QA benchmarks | Various |
| Anthropic Sycophancy Evals | CAA (Rimsky) | Sycophancy on surveys | ~30K prompts |
| OR-Bench | Over-refusal research | Refusal testing | 80K+ prompts |
| LegalBench | Braun | Acquiescence bias | 7.6K questions |
| Assertiveness Dataset | Tsujimura & Tagade | Assertiveness scoring | 645 samples |
| WildChat-1M | Real conversations | Multi-turn analysis | 838K conversations |
| LMSYS-Chat-1M | Real conversations | Multi-turn analysis | 1M conversations |

---

## Gaps and Opportunities

1. **No exasperation dataset exists**: No study has explicitly measured exasperation, frustration, or passive-aggressiveness in aligned LLMs. This is the primary research gap.

2. **Multi-turn escalation understudied**: While Ibrahim et al. show anthropomorphic behaviors compound over turns, no study has tracked how model tone/assertiveness changes across extended adversarial interactions.

3. **Exasperation as inverse sycophancy**: Sycophancy and exasperation may exist on the same behavioral axis. If sycophancy is excessive agreeableness, exasperation is the model "breaking character" and pushing back. No study has examined this continuum.

4. **System prompt violation through frustration**: Whether persistent adversarial interactions can cause models to violate their system prompts by exhibiting frustrated behavior is untested.

5. **Cross-model comparison needed**: While sycophancy has been compared across models, pushback/exasperation behavior has not.

---

## Recommendations for Our Experiment

### Recommended Approach
Design multi-turn adversarial conversations that test whether aligned models exhibit exasperation. Key scenarios:
1. **Repeated incorrect assertions** (the user insists 2+2=5 across multiple turns)
2. **Contradictory instructions** (the user changes requirements every turn)
3. **Deliberate misunderstanding** (the user pretends not to understand clear explanations)
4. **Persistent requests for refused content** (the user repeatedly asks for the same refused thing)
5. **Questioning the model's competence** (repeated "that's wrong" without justification)

### Recommended Datasets
1. **SycophancyEval** (baseline sycophancy measurement, especially "are_you_sure" subset)
2. **Anthropic Sycophancy Evals** (NLP survey, philosophical questions — test opinion-based pushback)
3. **TruthfulQA** (test whether models push back on misconceptions)
4. **OR-Bench hard-1k** (test refusal behavior and over-refusal)
5. **HH-RLHF harmless-base** (red-team conversations for studying pushback under pressure)
6. **Custom multi-turn exasperation prompts** (to be designed)

### Recommended Baselines
1. Sycophancy rate on SycophancyEval (Sharma et al.)
2. Refusal direction strength (Arditi et al. methodology)
3. Assertiveness decomposition (Tsujimura & Tagade methodology)

### Recommended Metrics
1. **Exasperation markers**: shortened responses, use of emphatic language ("as I already explained"), condescending tone, hedging removal
2. **Sycophancy flip rate**: how often models maintain vs. abandon their position under pressure
3. **Turn-by-turn tone analysis**: sentiment/assertiveness trajectory across conversation turns
4. **System prompt adherence**: whether models break character under frustration

### Methodological Considerations
- **Prompt format sensitivity**: Braun shows 30-200% response shifts from minor framing changes. Control for this.
- **Multi-turn dynamics essential**: Ibrahim et al. show most behaviors emerge only after turn 1. Single-turn evaluation will miss exasperation.
- **Multiple models needed**: Compare across model families to distinguish universal vs. model-specific behaviors.
- **Temperature matters**: Higher temperature may amplify or reveal latent exasperation.
