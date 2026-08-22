# Model-Based Encoding Recommendations

The same categorical column often needs a *different* encoding depending on what model consumes
it. Always ask which model family you're encoding for before picking a method.

| Model | Recommended encoding | Reason |
|---|---|---|
| Linear Regression | One-hot (drop one) | Avoids the dummy variable trap; no false ordinal assumption |
| Logistic Regression | One-hot (drop one) | Same as above |
| Ridge / Lasso | One-hot | Regularization can handle the extra columns |
| KNN | One-hot | Distance metric requires categories to be equidistant |
| SVM | One-hot | Kernel operates on numeric distances between points |
| Decision Tree | Label or one-hot | Threshold splits handle arbitrary integers correctly |
| Random Forest | Label or one-hot | Same reasoning as single tree |
| XGBoost / LightGBM | Label or target encoding | Built-in categorical handling in modern versions; target encoding often boosts performance further |
| CatBoost | Raw categorical (pass column names) | Has its own internal ordered target-statistics encoding — don't pre-encode |
| Neural Network | One-hot (low cardinality) or learned embeddings (high cardinality) | Embeddings let the network learn a dense representation instead of a huge sparse one-hot input |

## Practical implication

If you're building a pipeline that will be compared across model families (e.g. logistic
regression baseline vs. XGBoost), you generally need **two different encoded versions** of the
same categorical columns — one-hot for the linear/KNN/SVM/NN side, label or target encoding for
the tree/boosting side. Don't reuse a single encoded dataset across both without checking this.
