# Naive Bayes (Gaussian)

code: `code/naive_bayes.py` — classes `SklearnNaiveBayes`, `ScratchNaiveBayes`

## The Core Idea — Bayes' Theorem + Independence Assumption

Naive Bayes predicts the class that maximizes the posterior probability,
computed via Bayes' theorem, while **naively** assuming all features are
conditionally independent given the class.

```
P(class | features) ∝ P(class) * P(feature1 | class) * P(feature2 | class) * ...
```

This "naive" independence assumption is almost always technically false, but
the classifier works surprisingly well in practice, especially for
high-dimensional data like text.

## The Math — Step by Step (Gaussian variant)

**Step 1: Learn per-class, per-feature Gaussian parameters.**
For each class `c` and feature `j`: estimate `mean(c,j)` and `var(c,j)` from
training data.

**Step 2: Likelihood.**
```
P(x_j | class=c) = (1 / sqrt(2*pi*var(c,j))) * exp(-(x_j - mean(c,j))^2 / (2*var(c,j)))
```

**Step 3: Posterior (in log-space for numerical stability).**
```
log P(class=c | x) ∝ log P(class=c) + sum_j log P(x_j | class=c)
```

**Step 4: Predict** the class with the highest log-posterior.

## Key Hyperparameters

| Param | Effect |
|---|---|
| `var_smoothing` | Small value added to variances to avoid division by zero on near-constant features |
| `priors` | Optionally fix class priors instead of learning from data frequency |

## When to Use

✓ Fast baseline, especially for text classification (with Multinomial variant)
✓ High-dimensional data, small training sets
✓ Real-time / low-latency scoring
✓ Features are approximately independent given the class

## When NOT to Use

✗ Strongly correlated features (violates core assumption, biases probabilities)
✗ Need well-calibrated probabilities (NB tends toward overconfidence)

## Agent Metadata Summary

```python
SklearnNaiveBayes.METADATA
# family: probabilistic | interpretable: True | sensitive_to_scaling: False
# training_speed: fast | inference_speed: fast | handles_imbalance_well: True
```
