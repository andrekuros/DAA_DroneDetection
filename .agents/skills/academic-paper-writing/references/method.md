# Method

## Goal

Enough detail to reproduce the interface and training; motivation → design → technical advantage per subsection.

## Mini-outline per subsection

- **Motivation**: why this interface/module
- **Design**: inputs, architecture, outputs into classical solver
- **Advantage**: what it enables under the problem constraints (not marketing)

## Must include

- Input features / token definitions
- Network (layers, width, heads) or matched MLP size
- Output interface (priorities, locks, pair scores, …)
- Loss and update rule (batch size, warmup if relevant)
- Training horizon, validation metric, checkpoint rule
- Inference replan triggers
- Fairness: same mask / cadence as baselines

## Avoid

“Trained for hundreds of episodes” without specifics; stacking novelty adjectives; describing code filenames instead of algorithms.
