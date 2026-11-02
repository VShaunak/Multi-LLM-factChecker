# Mathematics used by v4 FREE NO WEB

## 1. Pairwise answer comparison

For every pair of models `i` and `j`:

```text
P(i,j) = (0.35T(i,j) + 0.20K(i,j) + 0.20F(i,j)) / 0.75
```

- `T`: TF-IDF cosine similarity of complete answers
- `K`: TF-IDF cosine similarity of extracted key points
- `F`: TF-IDF cosine similarity of checkable claims

The division by `0.75` renormalizes the three answer-only components to a 0-to-1 score.

Ten models create:

```text
10 × 9 / 2 = 45 pairs
```

## 2. Agreement

```text
C(i) = average of P(i,j) for all j ≠ i
```

## 3. Consensus correctness estimate

Without web evidence, the app calculates:

```text
Correctness(i) = 0.70 ClaimConsensus(i) + 0.30 NumberConsensus(i)
```

Claim consensus checks whether other models make semantically similar claims. Number consensus checks whether numbers in one response appear in at least one other response.

This is not external fact verification.

## 4. Quality

```text
Q(i) = 0.62 Correctness(i) + 0.23 Completeness(i) + 0.15 Clarity(i)
```

Completeness uses key-point and claim coverage. Clarity uses response length and sentence readability heuristics.

## 5. Confidence

Each model returns a confidence value from 0 to 1. The app applies softmax normalization and rescales the largest value to 1.

## 6. Speed

The fastest observed response receives a speed score near 1, and the slowest receives a score near 0.

## 7. Final score

Default weights:

```text
S(i) = 0.25C(i) + 0.55Q(i) + 0.05W(i) + 0.15V(i)
```

- `C`: agreement
- `Q`: quality
- `W`: normalized confidence
- `V`: observed speed

The model with the highest `S(i)` is selected. No extra judge LLM is used.
