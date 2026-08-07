# Paper Review (Adversarial)

Use at end of draft or when the user asks for publication readiness.

## Workflow

1. Read as a skeptical reviewer for the target venue.
2. Answer the five-dimension question list below in writing.
3. Mark each item: OK / risk / blocker.
4. Revise until every **blocker** and high **risk** is addressed or explicitly scoped out in Limitations.
5. Re-check Abstract/Intro claims against tables.

## Five dimensions

### 1. Contribution
- Is the contribution sentence falsifiable and matched by experiments?
- Are co-equal claims both evidenced, or is one decorative?
- Would a reviewer say “incremental engineering” — if so, what framing fixes it?

### 2. Writing clarity
- One message per paragraph? Topic sentences present?
- Terms defined before reuse? Notation stable?
- Does reverse outline map evidence → topic → thesis?

### 3. Experimental strength
- Fair baselines under the same information mask?
- Matched controls for architecture claims?
- Statistical protocol stated (paired, N, CI)?
- Honest negatives reported?

### 4. Evaluation completeness
- Ablations that a reviewer will demand already present?
- Sensitivity / boundary suite if metric is heuristic?
- Oracle / information upper bound interpreted correctly?

### 5. Method design soundness
- Interface to classical solver clear?
- Training objective aligned with evaluation score (or gap acknowledged)?
- Any confound (optimizer, features, pad size) that could flip a claim?

## Claim–evidence hard constraint

For Abstract and Introduction, list:

`Claim: ... | Evidence: ... | Status: supported/needs evidence`

Unsupported → weaken or remove before submission.
