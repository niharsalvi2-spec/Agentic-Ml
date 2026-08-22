# Classification Metrics

code: `code/classification_metrics.py`

## The Confusion Matrix — Foundation of Everything

```
Actual ->        Positive (1)        Negative (0)
Predicted v
Positive (1)    True Positive (TP)  False Positive (FP)  <- "Type I Error"
Negative (0)    False Negative (FN) True Negative (TN)     <- "Type II Error"
```

Memory trick: first word = was the prediction correct? second word = what did
the model predict?

- **TP**: actual sick, predicted sick (correct)
- **TN**: actual healthy, predicted healthy (correct)
- **FP**: actual healthy, predicted sick (false alarm)
- **FN**: actual sick, predicted healthy (missed case -- often the dangerous one)

`binary_counts(y_true, y_pred, positive_label)` returns all four as a dict.

## Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**The trap:** 100 patients, 90 healthy / 10 sick. A model that predicts
"healthy" for everyone gets `TP=0, TN=90, FP=0, FN=10` -> **90% accuracy**
while catching zero real cases. Accuracy alone is never enough on imbalanced
data.

## Precision

```
Precision = TP / (TP + FP)
```
"When the model raises an alarm, how often is it right?" Low precision =
many false alarms. Critical when false positives are costly (spam filters,
legal accusations, drug recommendations).

## Recall (Sensitivity, True Positive Rate)

```
Recall = TP / (TP + FN)
```
"Of all actual positives, how many did we catch?" Critical when false
negatives are costly (cancer screening, fraud detection, security screening).

**Rule of thumb:** missing a positive case is dangerous -> optimize Recall.
False alarms are expensive -> optimize Precision.

## Specificity (True Negative Rate)

```
Specificity = TN / (TN + FP)
```
"Of all actual negatives, how many did we correctly clear?" Recall and
Specificity trade off: a lower decision threshold catches more positives
(higher recall) but flags more negatives too (lower specificity).

## F1 Score

Harmonic mean of precision and recall -- punishes extreme imbalance between
the two much more than an arithmetic mean would:
```
F1 = 2 x (Precision x Recall) / (Precision + Recall)
```
`P=1.0, R=0.0` -> arithmetic mean = 0.5 (misleadingly good) but **F1 = 0**
(correctly bad). F1 is only high when BOTH precision and recall are high.

## F-Beta Score

Weighted version of F1 when one of precision/recall matters more:
```
F_beta = (1+beta^2) x (Precision x Recall) / (beta^2 x Precision + Recall)
```
- `beta = 1` -> standard F1 (equal weight)
- `beta = 2` -> recall weighted 2x more (cancer/fraud detection -- missing cases is costly)
- `beta = 0.5` -> precision weighted 2x more (spam filters -- false alarms are costly)

## ROC Curve and ROC-AUC

Classifiers output a probability; you pick a threshold (default 0.5) to turn
it into a label. The ROC curve shows what happens to **TPR** (= Recall) and
**FPR** (= FP/(FP+TN) = 1 - Specificity) as the threshold sweeps from 0 to 1.

- Perfect classifier -> curve hits the top-left corner (TPR=1, FPR=0)
- Random classifier -> diagonal line (TPR = FPR always)
- **AUC** = area under this curve, a threshold-independent summary

**Intuitive meaning:** AUC = the probability that a randomly chosen positive
sample gets a higher score than a randomly chosen negative sample.

| AUC | Interpretation |
|---|---|
| 0.5 | random -- no better than a coin flip |
| 0.6 | poor |
| 0.7 | fair |
| 0.8 | good |
| 0.9 | excellent |
| 0.95+ | outstanding |

**Caveat:** with severe class imbalance, ROC-AUC can look artificially high
because it's dominated by how well the (numerous) negatives are ranked. Use
PR-AUC instead when the positive class is rare.

## Precision-Recall Curve and PR-AUC (Average Precision)

Plots Precision vs. Recall as the threshold sweeps. On imbalanced data, a
random classifier scores around the positive base rate (a flat line), so
PR-AUC is much more sensitive to genuine positive-class performance than
ROC-AUC. **Rule of thumb:** if the positive class is under 10% of the data,
prefer PR-AUC.

## Multiclass Averaging

| Mode | How it works | Best for |
|---|---|---|
| `macro` | Compute metric per class, then plain average | Balanced classes, care about rare classes equally |
| `weighted` | Per-class metric, averaged weighted by class frequency | Accounts for imbalance, reflects overall distribution |
| `micro` | Pool all TP/FP/FN across classes first, then compute one metric | Dominated by the majority class; overall-prediction-pool view |

## Function Reference

```python
confusion_matrix(y_true, y_pred, labels=None)
binary_counts(y_true, y_pred, positive_label=1)
accuracy_score(y_true, y_pred)
precision_score(y_true, y_pred, average="binary", positive_label=1)
recall_score(y_true, y_pred, average="binary", positive_label=1)
specificity_score(y_true, y_pred, positive_label=1)
f1_score(y_true, y_pred, average="binary", positive_label=1)
fbeta_score(y_true, y_pred, beta, average="binary", positive_label=1)
roc_curve(y_true, y_score, positive_label=1)              # -> fpr, tpr, thresholds
roc_auc_score(y_true, y_score, positive_label=1)
precision_recall_curve(y_true, y_score, positive_label=1) # -> precision, recall, thresholds
average_precision_score(y_true, y_score, positive_label=1)
classification_report(y_true, y_pred, y_score=None, positive_label=1, average="binary")
```
