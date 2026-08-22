"""
Example: association_rules.py

Run with:  python3 example_association_rules.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from association_rules import apriori, generate_rules, format_rule, support, confidence, lift


TRANSACTIONS = [
    {"Bread", "Milk", "Eggs"},
    {"Bread", "Diapers", "Beer", "Eggs"},
    {"Milk", "Diapers", "Beer", "Cola"},
    {"Bread", "Milk", "Diapers", "Beer"},
    {"Bread", "Milk", "Diapers", "Cola"},
]


def apriori_worked_example():
    print("=" * 70)
    print("Example 1: Apriori on the classic 5-transaction basket (min_support=0.4)")
    print("=" * 70)

    frequent = apriori(TRANSACTIONS, min_support=0.4)
    print(f"Found {len(frequent)} frequent itemsets total:")
    by_size = {}
    for itemset, s in frequent.items():
        by_size.setdefault(len(itemset), []).append((itemset, s))

    for size in sorted(by_size):
        print(f"\n  L{size}:")
        for itemset, s in sorted(by_size[size], key=lambda kv: -kv[1]):
            print(f"    {set(itemset)}: support={s:.2f}")
    print()


def why_lift_matters_example():
    print("=" * 70)
    print("Example 2: why Confidence alone can mislead -- Bread->Milk vs Diapers->Beer")
    print("=" * 70)

    for antecedent, consequent in [({"Bread"}, {"Milk"}), ({"Diapers"}, {"Beer"})]:
        supp = support(antecedent | consequent, TRANSACTIONS)
        conf = confidence(antecedent, consequent, TRANSACTIONS)
        lft = lift(antecedent, consequent, TRANSACTIONS)
        a_name, c_name = next(iter(antecedent)), next(iter(consequent))
        print(f"{{{a_name}}} -> {{{c_name}}}: support={supp:.2f}  "
              f"confidence={conf:.2f}  lift={lft:.2f}")

    print("-> Bread->Milk has 75% confidence but lift < 1: milk is just "
          "generally popular, bread isn't really driving it.")
    print("   Diapers->Beer has lift > 1: this is a genuinely meaningful, "
          "actionable association.\n")


def full_rule_mining_example():
    print("=" * 70)
    print("Example 3: full pipeline -- frequent itemsets -> ranked rules")
    print("=" * 70)

    frequent = apriori(TRANSACTIONS, min_support=0.4)
    rules = generate_rules(frequent, TRANSACTIONS, min_confidence=0.6)

    print(f"{len(rules)} rules with confidence >= 0.6, top 5 by lift:")
    for r in rules[:5]:
        print(" ", format_rule(r))
    print()


def bigger_random_basket_example():
    print("=" * 70)
    print("Example 4: Apriori on a larger randomly generated basket")
    print("=" * 70)

    import random
    random.seed(0)
    catalog = ["Bread", "Milk", "Eggs", "Diapers", "Beer", "Cola",
               "Chips", "Salsa", "Butter", "Cheese"]
    transactions = []
    for _ in range(200):
        # bias a few co-occurring pairs to create real structure to discover
        basket = set(random.sample(catalog, k=random.randint(2, 5)))
        if "Chips" in basket and random.random() < 0.7:
            basket.add("Salsa")
        transactions.append(basket)

    frequent = apriori(transactions, min_support=0.15)
    rules = generate_rules(frequent, transactions, min_confidence=0.5)
    print(f"{len(frequent)} frequent itemsets, {len(rules)} rules found.")
    print("Top 5 rules by lift:")
    for r in rules[:5]:
        print(" ", format_rule(r))


if __name__ == "__main__":
    apriori_worked_example()
    why_lift_matters_example()
    full_rule_mining_example()
    bigger_random_basket_example()
