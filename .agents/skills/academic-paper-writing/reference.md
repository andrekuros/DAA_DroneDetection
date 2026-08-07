# Academic Paper Writing — Reference

Load when doing a deep revise, review, or phrase-level polish.
Section-specific guides live under [references/](references/).

## Claim language bank

### Supported positive result
- “Method A yields higher mean M than baseline B over N paired seeds; the paired bootstrap 95% CI for ΔM excludes zero.”
- “The improvement is statistically detectable under protocol P; absolute effect size is Δ̂ (CI).”

### Null / negative architecture result
- “Under the same observation interface and training budget, the attention encoder does not outperform a matched MLP control (CI for Δ includes zero).”
- “Gains relative to B are therefore not attributable to set attention alone.”

### Scoped non-claim
- “We do not claim superiority over the omniscient oracle, which is reported only as an information upper bound.”
- “Results are limited to simulation under scenario family S.”

## Informal → scholarly rewrites

| Informal | Scholarly |
|----------|-----------|
| Attention claim fails | No significant attention-specific gain versus the matched MLP |
| Beats Hungarian | Improves score relative to visibility-masked Local-Hungarian |
| Wrong inductive bias | Myopic complete-information matching is a poor inductive bias when … |
| Free lunch | Does not by itself yield gains under … |
| Stack / codebase jargon in Methods | Implementation details deferred to supplementary material / code release |
| We show X is amazing | We evaluate X and report … |

## Related work skeleton

For each theme paragraph:
1. One-sentence theme
2. 2–4 key citations with *what they did*
3. One sentence gap relative to this paper (“Unlike …, we …”)

Minimum viable clusters for multi-robot / learning papers:
- Classical assignment / market-based allocation
- Learning and MARL for task allocation
- Hybrid learning + combinatorial solvers
- Partial observability / communication / time windows (as relevant)

## Methods minimum viable content

- Input features / token definitions
- Network (layers, width, heads) or MLP size
- Output interface to the classical solver (priorities, locks, pair scores, …)
- Loss and update rule
- Exploration (if any)
- Training horizon, validation metric, early stopping / checkpoint rule
- Inference-time replan triggers

## Metrics

If weights are heuristic:
- State that they encode relative preference (e.g. miss cost vs on-time reward)
- Preferably add a one-paragraph sensitivity check or justify from scenario parameters

Never present an ad-hoc weighted sum as “the” objective without saying it is an evaluation score.

## Statistics

State explicitly:
- Paired vs unpaired
- Bootstrap vs t-test / Wilcoxon
- Number of resamples
- Multiple-comparison stance if many algorithms compared (at least acknowledge)

Do not say “significant” for CI-overlapping comparisons.

## Venue heuristics

| Venue type | Emphasize | Compress |
|------------|-----------|----------|
| IEEE RA-L / letters | One claim, clear figs, tight related work | Secondary suites → brief subsection |
| IEEE TAES / T-ASE | Systems framing, scenarios, baselines | Pure ML architecture novelty |
| IJRR / T-RO | Theory or hardware / strong empirics | Sim-only modest gains |
| Workshop | Early evidence, clear limitations | Overclaiming generality |

## Final pre-submission scan

- Title matches actual claim (not abandoned research questions)
- Abstract numbers match tables
- Every acronym expanded at first use
- Consistent notation across Problem / Methods / Metrics
- Limitations paragraph exists
- Acknowledgments / funding / ethics as required by venue


## Claim language bank

### Supported positive result
- “Method A yields higher mean M than baseline B over N paired seeds; the paired bootstrap 95% CI for ΔM excludes zero.”
- “The improvement is statistically detectable under protocol P; absolute effect size is Δ̂ (CI).”

### Null / negative architecture result
- “Under the same observation interface and training budget, the attention encoder does not outperform a matched MLP control (CI for Δ includes zero).”
- “Gains relative to B are therefore not attributable to set attention alone.”

### Scoped non-claim
- “We do not claim superiority over the omniscient oracle, which is reported only as an information upper bound.”
- “Results are limited to simulation under scenario family S.”

## Informal → scholarly rewrites

| Informal | Scholarly |
|----------|-----------|
| Attention claim fails | No significant attention-specific gain versus the matched MLP |
| Beats Hungarian | Improves score relative to visibility-masked Local-Hungarian |
| Wrong inductive bias | Myopic complete-information matching is a poor inductive bias when … |
| Free lunch | Does not by itself yield gains under … |
| Stack / codebase jargon in Methods | Implementation details deferred to supplementary material / code release |
| We show X is amazing | We evaluate X and report … |

## Related work skeleton

For each theme paragraph:
1. One-sentence theme
2. 2–4 key citations with *what they did*
3. One sentence gap relative to this paper (“Unlike …, we …”)

Minimum viable clusters for multi-robot / learning papers:
- Classical assignment / market-based allocation
- Learning and MARL for task allocation
- Hybrid learning + combinatorial solvers
- Partial observability / communication / time windows (as relevant)

## Methods minimum viable content

- Input features / token definitions
- Network (layers, width, heads) or MLP size
- Output interface to the classical solver (priorities, locks, pair scores, …)
- Loss and update rule
- Exploration (if any)
- Training horizon, validation metric, early stopping / checkpoint rule
- Inference-time replan triggers

## Metrics

If weights are heuristic:
- State that they encode relative preference (e.g. miss cost vs on-time reward)
- Preferably add a one-paragraph sensitivity check or justify from scenario parameters

Never present an ad-hoc weighted sum as “the” objective without saying it is an evaluation score.

## Statistics

State explicitly:
- Paired vs unpaired
- Bootstrap vs t-test / Wilcoxon
- Number of resamples
- Multiple-comparison stance if many algorithms compared (at least acknowledge)

Do not say “significant” for CI-overlapping comparisons.

## Venue heuristics

| Venue type | Emphasize | Compress |
|------------|-----------|----------|
| IEEE RA-L / letters | One claim, clear figs, tight related work | Secondary suites → brief subsection |
| IEEE TAES / T-ASE | Systems framing, scenarios, baselines | Pure ML architecture novelty |
| IJRR / T-RO | Theory or hardware / strong empirics | Sim-only modest gains |
| Workshop | Early evidence, clear limitations | Overclaiming generality |

## Final pre-submission scan

- Title matches actual claim (not abandoned research questions)
- Abstract numbers match tables
- Every acronym expanded at first use
- Consistent notation across Problem / Methods / Metrics
- Limitations paragraph exists
- Acknowledgments / funding / ethics as required by venue
