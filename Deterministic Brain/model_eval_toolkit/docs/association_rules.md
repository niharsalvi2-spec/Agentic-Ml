# Association Rule Mining

code: `code/association_rules.py`

Goal: discover interesting relationships between items across many
transactions — most famously, Market Basket Analysis ("customers who buy
diapers also tend to buy beer").

## Frequent Itemsets and Support

An **itemset** is any collection of one or more items. **Support** measures
how often an itemset appears across all transactions:

```
Support(X) = (transactions containing X) / (total transactions)
```

An itemset is **frequent** if `Support(X) >= min_support` (a threshold you
choose).

## Association Rules — Support, Confidence, Lift

A rule `X -> Y` ("antecedent -> consequent") says: transactions containing X
tend to also contain Y.

| Metric | Formula | Answers |
|---|---|---|
| Support | `Support(X union Y)` | How common is this pattern overall? |
| Confidence | `Support(X union Y) / Support(X)` = `P(Y \| X)` | Given X, how often does Y also occur? |
| Lift | `Confidence(X->Y) / Support(Y)` = `Support(X union Y) / (Support(X) * Support(Y))` | Is X really *driving* Y, or is Y just generally popular anyway? |

**Why Confidence alone is misleading:** `{Bread} -> {Milk}` can show 90%
confidence purely because milk is bought by ~90% of *all* customers
regardless of bread — bread isn't actually influencing anything. Lift
corrects for this baseline popularity.

**Interpreting Lift:**

| Lift | Meaning |
|---|---|
| = 1 | X and Y are independent — X tells you nothing extra about Y |
| > 1 | Positively associated — genuinely useful, actionable pattern |
| < 1 | Negatively associated — possibly substitute products |

**Best practice:** filter by minimum Support (common enough to matter),
filter by minimum Confidence (reliable enough), then rank by Lift to surface
the genuinely interesting rules.

## The Apriori Algorithm

**The problem:** with just 100 products there are `2^100` possible itemsets
— checking all of them is impossible.

**The Apriori Principle:** *if an itemset is infrequent, all of its
supersets are also infrequent* (adding items to a set can only keep or
shrink the number of transactions containing it, never grow it). This lets
you prune entire branches of the search space without ever computing their
support.

**Algorithm outline:**
1. `C1`: every individual item → compute support → keep frequent ones as `L1`
2. `C2`: join pairs of items from `L1` → compute support → keep as `L2`
3. `Ck`: join `(k-1)`-itemsets from `L(k-1)` that share their first `k-2`
   items, **but only keep a candidate if every one of its `(k-1)`-subsets is
   already frequent** (the core prune) → compute support → keep as `Lk`
4. Repeat until no new frequent itemsets are found
5. Output: `L1 ∪ L2 ∪ L3 ∪ ...`

**Limitations:** repeated full database scans (one per itemset size), a
candidate-generation explosion if `min_support` is set too low, and a
`min_support` value that's genuinely hard to pick (too high misses rare
but important patterns; too low produces a flood of noisy rules). FP-Growth
and ECLAT are faster modern alternatives that avoid explicit candidate
generation.

## From Frequent Itemsets to Rules

For every frequent itemset of size >= 2, generate every non-trivial split
into `antecedent -> consequent` (every non-empty proper subset as the
antecedent, the remaining items as the consequent), then compute
Confidence and Lift for each and keep the ones meeting a minimum confidence.

## Real-World Applications

Retail product placement and bundle recommendations, "customers who bought
X also bought Y" recommenders, web-navigation optimization, early-diagnosis
symptom-combination patterns in healthcare, gene co-occurrence research,
and fraud-pattern detection.

## Function Reference

```python
support(itemset, transactions)
apriori(transactions, min_support)               # -> {itemset: support}
confidence(antecedent, consequent, transactions)
lift(antecedent, consequent, transactions)
generate_rules(frequent_itemsets, transactions, min_confidence=0.0)
    # -> [{"antecedent", "consequent", "support", "confidence", "lift"}, ...]
    # sorted by lift, descending
format_rule(rule)   # human-readable one-line summary
```
