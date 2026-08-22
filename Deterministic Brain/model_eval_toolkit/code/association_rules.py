"""
association_rules.py
-----------------------
Pure-Python (itertools + collections, no external deps) implementation of
frequent itemset mining (Apriori algorithm) and association rule generation
with Support, Confidence, and Lift.

Typical usage:

    transactions = [
        {"Bread", "Milk", "Eggs"},
        {"Bread", "Diapers", "Beer", "Eggs"},
        {"Milk", "Diapers", "Beer", "Cola"},
        {"Bread", "Milk", "Diapers", "Beer"},
        {"Bread", "Milk", "Diapers", "Cola"},
    ]
    frequent_itemsets = apriori(transactions, min_support=0.4)
    rules = generate_rules(frequent_itemsets, transactions, min_confidence=0.6)
"""

from itertools import combinations


# --------------------------------------------------------------------------
# Support
# --------------------------------------------------------------------------

def support(itemset, transactions):
    """
    Support(X) = (# transactions containing X) / (total transactions).
    itemset: any iterable of items (will be treated as a set).
    transactions: iterable of sets (or iterables) of items.
    """
    itemset = frozenset(itemset)
    if not transactions:
        return 0.0
    count = sum(1 for t in transactions if itemset <= set(t))
    return count / len(transactions)


# --------------------------------------------------------------------------
# Apriori algorithm — frequent itemset mining
# --------------------------------------------------------------------------

def _generate_candidates(prev_frequent, k):
    """
    Generate candidate k-itemsets from frequent (k-1)-itemsets by joining
    pairs that share the first k-2 items (standard Apriori join step), then
    apply the Apriori prune: discard a candidate if any of its (k-1)-subsets
    is not itself frequent.
    """
    prev_list = sorted(prev_frequent, key=lambda s: sorted(s))
    prev_set = set(prev_frequent)
    candidates = set()

    for i in range(len(prev_list)):
        for j in range(i + 1, len(prev_list)):
            a, b = sorted(prev_list[i]), sorted(prev_list[j])
            if k > 2 and a[:-1] != b[:-1]:
                continue  # only join itemsets sharing all but the last item
            union = frozenset(prev_list[i] | prev_list[j])
            if len(union) != k:
                continue
            # Apriori prune: every (k-1)-subset of the candidate must be frequent
            if all(frozenset(sub) in prev_set for sub in combinations(union, k - 1)):
                candidates.add(union)

    return candidates


def apriori(transactions, min_support):
    """
    Runs the full Apriori algorithm.

    transactions: iterable of sets (or iterables) of items.
    min_support: minimum support threshold (0-1) an itemset must meet to be
                 considered "frequent".

    Returns a dict {itemset (frozenset): support (float)} covering all
    frequent itemsets of every size found (L1 union L2 union L3 ...).
    """
    transactions = [set(t) for t in transactions]
    n = len(transactions)
    if n == 0:
        return {}

    # C1: every individual item that appears in any transaction
    all_items = set()
    for t in transactions:
        all_items |= t
    candidates_1 = {frozenset([item]) for item in all_items}

    frequent_itemsets = {}
    current_level = {}
    for c in candidates_1:
        s = support(c, transactions)
        if s >= min_support:
            current_level[c] = s

    frequent_itemsets.update(current_level)
    k = 2

    while current_level:
        candidates_k = _generate_candidates(set(current_level.keys()), k)
        current_level = {}
        for c in candidates_k:
            s = support(c, transactions)
            if s >= min_support:
                current_level[c] = s
        frequent_itemsets.update(current_level)
        k += 1

    return frequent_itemsets


# --------------------------------------------------------------------------
# Rule generation — Confidence and Lift
# --------------------------------------------------------------------------

def confidence(antecedent, consequent, transactions):
    """
    Confidence(X -> Y) = Support(X union Y) / Support(X) = P(Y | X).
    """
    antecedent = frozenset(antecedent)
    consequent = frozenset(consequent)
    union = antecedent | consequent
    supp_x = support(antecedent, transactions)
    if supp_x == 0:
        return 0.0
    return support(union, transactions) / supp_x


def lift(antecedent, consequent, transactions):
    """
    Lift(X -> Y) = Support(X union Y) / (Support(X) * Support(Y))
                 = Confidence(X -> Y) / Support(Y)

    lift == 1 -> X and Y independent (rule not meaningful)
    lift > 1  -> positively associated (genuinely interesting rule)
    lift < 1  -> negatively associated (possible substitute products)
    """
    antecedent = frozenset(antecedent)
    consequent = frozenset(consequent)
    supp_y = support(consequent, transactions)
    if supp_y == 0:
        return 0.0
    return confidence(antecedent, consequent, transactions) / supp_y


def generate_rules(frequent_itemsets, transactions, min_confidence=0.0):
    """
    From a dict of frequent itemsets (as returned by apriori()), generate
    every possible non-trivial rule X -> Y for each itemset of size >= 2
    (every non-empty proper subset X, with Y = itemset - X), compute
    Support/Confidence/Lift for each, and keep only rules meeting
    min_confidence.

    Returns a list of dicts, each:
        {"antecedent": frozenset, "consequent": frozenset,
         "support": float, "confidence": float, "lift": float}
    sorted by lift, descending.
    """
    rules = []
    for itemset, itemset_support in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        for r in range(1, len(items)):
            for antecedent_items in combinations(items, r):
                antecedent = frozenset(antecedent_items)
                consequent = itemset - antecedent
                if not consequent:
                    continue
                conf = confidence(antecedent, consequent, transactions)
                if conf >= min_confidence:
                    rules.append({
                        "antecedent": antecedent,
                        "consequent": consequent,
                        "support": itemset_support,
                        "confidence": conf,
                        "lift": lift(antecedent, consequent, transactions),
                    })

    rules.sort(key=lambda r: r["lift"], reverse=True)
    return rules


def format_rule(rule):
    """Human-readable one-line summary of a rule dict from generate_rules()."""
    ant = ", ".join(sorted(rule["antecedent"]))
    cons = ", ".join(sorted(rule["consequent"]))
    return (f"{{{ant}}} -> {{{cons}}}  "
            f"(support={rule['support']:.2f}, confidence={rule['confidence']:.2f}, "
            f"lift={rule['lift']:.2f})")


if __name__ == "__main__":
    transactions = [
        {"Bread", "Milk", "Eggs"},
        {"Bread", "Diapers", "Beer", "Eggs"},
        {"Milk", "Diapers", "Beer", "Cola"},
        {"Bread", "Milk", "Diapers", "Beer"},
        {"Bread", "Milk", "Diapers", "Cola"},
    ]
    frequent = apriori(transactions, min_support=0.4)
    print(f"Found {len(frequent)} frequent itemsets:")
    for itemset, s in sorted(frequent.items(), key=lambda kv: (len(kv[0]), -kv[1])):
        print(f"  {set(itemset)}: support={s:.2f}")

    rules = generate_rules(frequent, transactions, min_confidence=0.6)
    print(f"\n{len(rules)} rules with confidence >= 0.6, sorted by lift:")
    for r in rules:
        print(" ", format_rule(r))
