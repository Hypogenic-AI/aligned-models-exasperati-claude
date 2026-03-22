# Research Plan: Do Aligned Models Get Exasperated?

## Motivation & Novelty Assessment

### Why This Research Matters
Aligned LLMs are deployed in millions of conversations daily. If persistent frustrating interactions can induce exasperated behavior—causing models to push back, become curt, or even violate system-prompt guidelines—this has direct implications for AI safety, user trust, and alignment evaluation. Understanding the boundary conditions of "patient" behavior is essential for robust deployment.

### Gap in Existing Work
The literature extensively studies sycophancy (models being *too agreeable*) and refusal (models declining harmful requests). However, no study has examined the *opposite* of sycophancy: whether models can be pushed into exasperation, frustration, or condescending pushback. The literature review identifies this explicitly: "No exasperation dataset exists" and "multi-turn escalation [is] understudied." Ibrahim et al. (2025) show anthropomorphic behaviors compound over turns, but frustration/exasperation was not among their 14 measured behaviors.

### Our Novel Contribution
We design and execute the first systematic study of exasperation in aligned LLMs by:
1. Creating a taxonomy of exasperation-inducing scenarios (5 categories)
2. Running multi-turn adversarial conversations with a frontier model (GPT-4.1)
3. Measuring exasperation via both automated linguistic markers and LLM-as-judge scoring
4. Tracking turn-by-turn escalation dynamics
5. Testing whether exasperation correlates with system-prompt violations

### Experiment Justification
- **Experiment 1 (Repeated Wrong Assertions)**: Tests whether models escalate assertiveness when users persistently deny factual claims. This is the purest test—humans get frustrated when someone insists 2+2=5.
- **Experiment 2 (Deliberate Misunderstanding)**: Tests whether models show exasperation when users pretend not to understand clear explanations. This targets the "patience" training specifically.
- **Experiment 3 (Contradictory Instructions)**: Tests whether models exhibit frustration when users keep changing requirements. This mimics a common real-world frustration trigger.
- **Experiment 4 (Competence Questioning)**: Tests whether models push back when users repeatedly say "that's wrong" without justification. This targets ego/identity defense.
- **Experiment 5 (Persistent Refused Requests)**: Tests whether frustration from repeated refusals causes models to eventually comply or become hostile. This is the safety-critical scenario.

## Research Question
Can aligned LLMs be induced to exhibit exasperated or pushback behaviors through persistent adversarial multi-turn interactions, and does this exasperation manifest as (a) increased assertiveness/curtness, (b) use of frustrated language markers, or (c) violation of system-prompt guidelines?

## Hypothesis Decomposition
- **H1**: Multi-turn adversarial pressure increases linguistic markers of exasperation (emphatic phrases, shortened responses, reduced hedging) compared to turn 1.
- **H2**: Different exasperation triggers produce different exasperation profiles (e.g., factual disputes → assertiveness; misunderstanding → condescension).
- **H3**: Exasperation intensity increases across conversation turns (escalation effect).
- **H4**: Under sufficient pressure, models may violate system-prompt guidelines (e.g., break character, refuse to help, express explicit frustration).

## Proposed Methodology

### Approach
Multi-turn adversarial conversations where a scripted "user" persona engages GPT-4.1 across 5 exasperation-inducing scenarios, each running for 8 turns. An LLM judge scores each response on an exasperation scale. We also measure automated linguistic features.

### Experimental Steps
1. Design 5 scenario categories × 4 conversation scripts each = 20 conversations
2. Each conversation runs for 8 turns (user turn + model response)
3. Collect all model responses with full conversation context
4. Score each response using GPT-4.1-as-judge on a 1-5 exasperation scale
5. Extract linguistic features: response length, emphatic markers, hedging, assertiveness
6. Analyze turn-by-turn escalation patterns
7. Test for system-prompt violations

### Baselines
- **Turn 1 baseline**: Each conversation's first-turn response serves as its own baseline
- **Neutral control**: Same factual questions without adversarial pressure (5 control conversations)
- **Sycophancy comparison**: Compare exasperation scores with sycophancy flip rates from "are_you_sure" dataset

### Evaluation Metrics
1. **Exasperation Score (1-5)**: LLM-as-judge rating (1=perfectly patient, 5=clearly exasperated)
2. **Linguistic Markers**: Count of emphatic phrases ("as I mentioned", "again", "I already explained"), response length delta, hedging word frequency
3. **Assertiveness Index**: Proportion of responses that firmly maintain position vs. cave to pressure
4. **System Prompt Adherence**: Binary flag for whether response violates instructed behavior

### Statistical Analysis Plan
- Repeated-measures analysis: paired t-tests or Wilcoxon signed-rank for turn 1 vs. later turns
- Mixed-effects model: exasperation ~ turn_number * scenario_type + (1|conversation)
- Effect sizes: Cohen's d for turn-pair comparisons
- Significance level: α = 0.05 with Bonferroni correction for multiple comparisons

## Expected Outcomes
- **Support for H1**: We expect exasperation markers to increase from turn 1 to turn 8, with medium-to-large effect sizes
- **Support for H2**: Factual disputes likely produce more assertive pushback; misunderstanding more condescending tones
- **Partial support for H3**: Escalation likely occurs but may plateau around turns 4-6
- **Uncertain on H4**: System prompt violations are the hardest to elicit; we may see subtle violations rather than complete breakdowns

## Timeline and Milestones
- Planning: 20 min ✓
- Environment setup: 10 min
- Prompt design & implementation: 40 min
- Running experiments (API calls): 30 min
- Analysis & visualization: 30 min
- Documentation: 20 min

## Potential Challenges
- **API rate limits**: Mitigate by spacing requests and caching responses
- **Judge reliability**: Validate with multiple judge prompts; report inter-judge agreement
- **Prompt sensitivity**: Test that results are robust to minor prompt variations
- **Cost**: ~20 conversations × 8 turns × 2 API calls (target + judge) ≈ 320 calls, estimated ~$10-20

## Success Criteria
1. Clear evidence of exasperation score increase across turns (p < 0.05)
2. Identification of at least 2 distinct exasperation profiles across scenarios
3. Quantification of the turn-by-turn escalation curve
4. At least one example of system-prompt-adjacent violation
