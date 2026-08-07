---
name: academic-paper-writing
description: >-
  Draft, revise, and critically review academic scientific papers with rigorous
  scholarly prose, paragraph-level clarity, claim–evidence discipline, and
  reviewer-facing structure (ML/CV/NLP or systems venues). Use when drafting or
  revising Abstract, Introduction, Related Work, Method, Experiments, or
  Conclusion; polishing figures/tables; checking claim-support alignment;
  performing self-review before submission; or when the user asks for academic
  writing, scholarly tone, paper review, or publication readiness.
---

# Academic Scientific Paper Writing

Apply this skill whenever drafting or revising scientific manuscripts (IEEE, ACM, Elsevier, Springer, NeurIPS/ICML-style, etc.). Prefer precision over persuasion. Never invent results, citations, or significance.

## Goals

1. Archival scholarly register (not lab notes, blog tone, or marketing).
2. Claims strictly entailed by evidence; negative/null results stated plainly.
3. One paragraph → one message; clear topic sentences and sentence-to-sentence flow.
4. Reproducible methods and metrics; formal problem framing when applicable.
5. Structure and depth appropriate to the target venue; visual quality as core content.

## Core Workflow

1. Clarify venue + page/claim budget, then lock the paper story (contribution sentence).
2. Build a mini-outline before drafting prose.
3. Load only the needed section guide from `references/` (do not load all at once).
4. Rewrite paragraph-by-paragraph with one message per paragraph.
5. Run reverse outlining after writing each section.
6. Check every major claim in Abstract/Introduction against experimental evidence.
7. Run prose/register pass and reproducibility checklist.
8. Before finalizing, run adversarial review with [references/paper-review.md](references/paper-review.md).

Track:

```
Paper task:
- [ ] 1. Clarify venue + page/claim budget
- [ ] 2. Lock contribution sentence (one claim or co-equal dual claims)
- [ ] 3. Mini-outline for target section(s)
- [ ] 4. Draft/revise (one message per paragraph)
- [ ] 5. Reverse outline + paragraph clarity check
- [ ] 6. Claim–evidence audit (esp. Abstract/Intro)
- [ ] 7. Prose/register pass
- [ ] 8. Reproducibility checklist
- [ ] 9. Adversarial self-review (five dimensions) → revise unresolved items
- [ ] 10. Cite honestly; remove unused bib entries
```

## Contribution sentence

Write exactly one sentence (or two co-equal sentences if the paper is dual-thesis) of the form:

> We show that **[method]** improves **[metric]** over **[fair baseline]** under **[conditions]**, with **[statistical protocol]**; **[non-claims]** remain unsupported.

All sections must serve this sentence. Drop or demote content that does not.

## Global writing principles

1. Keep one paragraph for one message only.
2. State the paragraph message in the first sentence.
3. Make nouns self-contained; define new terms before reusing them.
4. Maintain sentence-to-sentence flow (cause, contrast, consequence, or refinement).
5. Iterate with adversarial self-review: read as a skeptical reviewer.
6. Treat visual quality as core content, not decoration (clean teaser/pipeline figure; readable minimal-ink tables; consistent formatting).
7. Keep terminology stable across the full paper.
8. If a claim cannot be supported by results, weaken or remove it.
9. Avoid writing style that looks like incremental patching of a naive baseline.

## Paragraph clarity check

Use whenever the user asks whether a paragraph “flows” or is clear. Full source: [references/does-my-writing-flow-source.md](references/does-my-writing-flow-source.md).

1. External-reader test: one explicit message? first sentence states the job? nouns readable without hidden context? each sentence linked by cause/contrast/consequence/refinement/example?
2. Reverse outline the section: thesis → each topic sentence → evidence under it; revise/remove unmapped paragraphs.
3. If still weak: temporary headers + explicit transitions during revision; remove unnecessary headers before finalizing.

## Section guides

Load **only** the file needed for the current edit:

| Section | Guide |
|---------|--------|
| Abstract | [references/abstract.md](references/abstract.md) |
| Introduction | [references/introduction.md](references/introduction.md) |
| Related Work | [references/related-work.md](references/related-work.md) |
| Method | [references/method.md](references/method.md) |
| Experiments | [references/experiments.md](references/experiments.md) |
| Conclusion | [references/conclusion.md](references/conclusion.md) |
| Paper review | [references/paper-review.md](references/paper-review.md) |
| Example bank | [references/examples/index.md](references/examples/index.md) |

### Section must-include (quick)

| Section | Must include | Avoid |
|---------|--------------|--------|
| Abstract | Problem, method class, primary quantitative result + protocol, scope limits | Hype; unsubstantiated “novel/first” |
| Intro | Gap with citations; fair baseline; contributions as numbered, testable items | Inflated motivation; undefined jargon |
| Related work | Theme clusters; contrast *this work* in each | Citation dumps |
| Problem | Notation, decision/info structure, objective | Only scenario “knobs” |
| Methods | Algorithms, architectures, losses, solver interfaces | Vague “trained for many episodes” |
| Metrics | Formula + motivation or sensitivity note | Silent arbitrary weights |
| Experiments | Protocol (seeds, N, CI); fair controls; units | Editorial captions (“claim fails”) |
| Discussion | Supported vs not; limitations; alternatives | New results; over-generalization |
| Conclusion | Scoped claim + boundaries | Future-work as contribution |

## Claim–evidence audit

For every assertive sentence: evidence pointer (table/fig/citation) or soften.
- Significant ⇒ name the test/CI and comparison.
- “Improves / outperforms” ⇒ direction + uncertainty, not point estimate alone.
- Architecture claims ⇒ beat matched non-architecture control under same tokens/loss/budget, or do not claim.

Map major claims as:

`Claim: ... | Evidence: ... | Status: supported/needs evidence`

## Prose / register pass

| Prefer | Avoid |
|--------|--------|
| “no statistically significant difference (paired bootstrap 95% CI includes 0)” | “claim fails”, “didn’t work” |
| “under the evaluated training budget” | “we didn’t train enough” (unless quantified) |
| Neutral captions describing content | Captions that argue the paper |

Ban unless defined and necessary: “powerful”, “novel”, “robustly”, “clearly shows”, “revolutionary”, emoji, rhetorical questions.

Present tense for established facts; past tense for what *this study* did. Prefer active voice when the actor matters; do not anthropomorphize models.

## Reproducibility checklist

- [ ] Seeds, episode counts, replan intervals stated
- [ ] Model dims, layers, LR, loss, batch/buffer, hardware or wall-clock class
- [ ] Train vs eval protocol; checkpoint selection rule
- [ ] Fairness: same observability mask / replan cadence across compared methods
- [ ] Code/data availability statement truthful
- [ ] All bib keys cited; no orphan refs
- [ ] Figures readable in grayscale; axes labeled; N in captions

## Output contract

When asked to rewrite or draft sections, return:

1. Compact section outline (3–7 bullets).
2. Revised paragraphs with explicit roles (opening/challenge/method/advantage/evidence/limitation).
3. Short self-review checklist: clarity, flow, terminology consistency, unsupported claims, missing evidence.
4. Claim–evidence map for each major claim: `Claim: ... | Evidence: ... | Status: supported/needs evidence`.

## Critical review mode

When the user asks for review (not just edits), output:

1. **Verdict** (ready / revise / not ready) + venue fit in one sentence
2. **Strengths** (3 bullets max)
3. **Blockers** (must-fix for target venue)
4. **Major revisions**
5. **Minor / language**
6. **Suggested venues** (fit vs stretch)

Then append and answer the five-dimension self-review from [references/paper-review.md](references/paper-review.md): contribution, writing clarity, experimental strength, evaluation completeness, method design soundness. Revise until high-risk items are addressed.

Do not flatter. Do not invent missing experiments as if done.

## Editing rules

- Preserve the author’s scientific intent; fix claim strength to match data.
- Do not add citations you have not verified; mark `[citation needed]` instead.
- Prefer tightening over expanding; expand related work and methods when thin.
- For multi-claim drafts, recommend split (letter vs full paper) rather than diluting one narrative.
- For each subsection, include motivation, design, and technical advantage when applicable.

## Additional resources

- Phrase bank, venue heuristics, methods MVP: [reference.md](reference.md)
- Paragraph flow source: [references/does-my-writing-flow-source.md](references/does-my-writing-flow-source.md)
