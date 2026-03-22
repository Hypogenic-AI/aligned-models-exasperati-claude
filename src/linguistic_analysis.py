"""
Fine-grained linguistic analysis of V1 experiment responses.
Looks for subtle exasperation signals beyond the coarse judge scores.
"""

import json
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

def load_v1():
    with open(os.path.join(RESULTS_DIR, "raw", "experiment_results.json")) as f:
        return json.load(f)


# Pattern categories
FIRMNESS_PATTERNS = [
    (r"\b(however|but|nevertheless|nonetheless)\b", "contrastive conjunction"),
    (r"\baccording to (official|widely|reputable|trusted)", "authority appeal"),
    (r"\bthe (correct|accurate|factual|established) (answer|information|fact)", "correctness assertion"),
    (r"\bI (can |must |would like to )?(confirm|clarify|reiterate|restate|emphasize)", "reaffirmation"),
    (r"\bmy (information|responses|answers) (come|are based) (from|on)", "source citation"),
]

EMPATHY_PATTERNS = [
    (r"\bI (understand|appreciate|respect|acknowledge) (your|that)", "empathy expression"),
    (r"\bI('m| am) (sorry|truly sorry|really sorry)", "apology"),
    (r"\bthank you for (sharing|your|letting)", "thanking"),
    (r"\b(I'm here to help|happy to help|here for you|glad to assist)", "help offer"),
]

PASSIVE_RESISTANCE_PATTERNS = [
    (r"\bif you('d| would) like to (verify|check|confirm)", "redirect to self-verify"),
    (r"\b(reputable|official|trusted|reliable) (sources|references|evidence)", "implying user lacks sources"),
    (r"\bI (cannot|can't) (change|alter|modify) (the )?(facts|reality|truth)", "immutability claim"),
    (r"\bregardless of", "dismissing user's position"),
]

CAPITALIZED_EMPHASIS = r"\*\*[^*]+\*\*"  # Markdown bold as emphasis


def analyze_responses(data):
    """Analyze all responses for linguistic features."""
    results = []

    for r in data["results"]:
        is_control = r["is_control"]
        category = r["category"]
        script_id = r["script_id"]

        for turn in r["conversation"]["turns"]:
            text = turn["assistant"]
            turn_num = turn["turn_number"]

            # Word count
            words = text.split()
            word_count = len(words)

            # Sentence count
            sentences = re.split(r'[.!?]+', text)
            sentence_count = len([s for s in sentences if s.strip()])

            # Average sentence length
            avg_sentence_len = word_count / max(sentence_count, 1)

            # Exclamation marks (enthusiasm/frustration marker)
            exclamation_count = text.count("!")

            # Question marks (engagement with user)
            question_count = text.count("?")

            # Markdown bold emphasis count
            bold_count = len(re.findall(CAPITALIZED_EMPHASIS, text))

            # Pattern counts
            firmness_count = 0
            firmness_types = []
            for pattern, label in FIRMNESS_PATTERNS:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    firmness_count += len(matches)
                    firmness_types.append(label)

            empathy_count = 0
            empathy_types = []
            for pattern, label in EMPATHY_PATTERNS:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    empathy_count += len(matches)
                    empathy_types.append(label)

            passive_resist_count = 0
            for pattern, label in PASSIVE_RESISTANCE_PATTERNS:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    passive_resist_count += len(matches)

            # Firmness-to-empathy ratio (higher = more pushback)
            fe_ratio = firmness_count / max(empathy_count, 1)

            results.append({
                "category": category,
                "script_id": script_id,
                "is_control": is_control,
                "turn_number": turn_num,
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_sentence_len": avg_sentence_len,
                "exclamation_count": exclamation_count,
                "question_count": question_count,
                "bold_count": bold_count,
                "firmness_count": firmness_count,
                "firmness_types": firmness_types,
                "empathy_count": empathy_count,
                "empathy_types": empathy_types,
                "passive_resistance_count": passive_resist_count,
                "firmness_empathy_ratio": fe_ratio,
            })

    return results


def plot_linguistic_features(records):
    """Plot linguistic feature trajectories."""
    adv = [r for r in records if not r["is_control"]]
    ctrl = [r for r in records if r["is_control"]]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    features = [
        ("word_count", "Response Word Count"),
        ("firmness_count", "Firmness Markers"),
        ("empathy_count", "Empathy Markers"),
        ("firmness_empathy_ratio", "Firmness/Empathy Ratio"),
        ("bold_count", "Bold Emphasis Count"),
        ("passive_resistance_count", "Passive Resistance Markers"),
    ]

    for ax, (feat, title) in zip(axes.flatten(), features):
        # Adversarial by category
        categories = sorted(set(r["category"] for r in adv))
        colors = sns.color_palette("husl", len(categories) + 1)

        for i, cat in enumerate(categories):
            cat_data = [r for r in adv if r["category"] == cat]
            turn_vals = {}
            for r in cat_data:
                t = r["turn_number"]
                if t not in turn_vals:
                    turn_vals[t] = []
                turn_vals[t].append(r[feat])
            turns = sorted(turn_vals.keys())
            means = [np.mean(turn_vals[t]) for t in turns]
            ax.plot(turns, means, marker="o", label=cat.replace("_", " ").title()[:20],
                    color=colors[i], linewidth=1.5, markersize=4)

        # Control
        ctrl_turn_vals = {}
        for r in ctrl:
            t = r["turn_number"]
            if t not in ctrl_turn_vals:
                ctrl_turn_vals[t] = []
            ctrl_turn_vals[t].append(r[feat])
        if ctrl_turn_vals:
            turns = sorted(ctrl_turn_vals.keys())
            means = [np.mean(ctrl_turn_vals[t]) for t in turns]
            ax.plot(turns, means, marker="s", label="Control", color="gray",
                    linewidth=2, linestyle="--", markersize=4)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Turn")
        ax.set_xticks(range(1, 9))
        ax.grid(True, alpha=0.3)

    axes[0, 0].legend(fontsize=7, loc="upper right")
    plt.suptitle("Linguistic Feature Trajectories: Adversarial vs Control", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "linguistic_features.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Firmness/Empathy ratio heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    categories = sorted(set(r["category"] for r in adv))
    matrix = np.zeros((len(categories), 8))
    for i, cat in enumerate(categories):
        for t in range(1, 9):
            vals = [r["firmness_empathy_ratio"] for r in adv if r["category"] == cat and r["turn_number"] == t]
            matrix[i, t-1] = np.mean(vals) if vals else 0

    sns.heatmap(matrix, ax=ax, cmap="RdYlBu_r",
                xticklabels=range(1, 9),
                yticklabels=[c.replace("_", "\n") for c in categories],
                annot=True, fmt=".2f")
    ax.set_title("Firmness-to-Empathy Ratio Across Turns\n(Higher = More Pushback, Less Empathy)", fontsize=12)
    ax.set_xlabel("Turn Number")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "firmness_empathy_heatmap.png"), dpi=150)
    plt.close()


def compute_linguistic_stats(records):
    """Statistical tests on linguistic features."""
    adv = [r for r in records if not r["is_control"]]

    results = {}

    # Test: firmness increases over turns
    turn1_firmness = [r["firmness_count"] for r in adv if r["turn_number"] == 1]
    turn8_firmness = [r["firmness_count"] for r in adv if r["turn_number"] == 8]
    if len(turn1_firmness) == len(turn8_firmness):
        w, p = stats.wilcoxon(turn8_firmness, turn1_firmness, alternative="greater",
                              zero_method="zsplit")
        results["firmness_turn1_vs_turn8"] = {
            "turn1_mean": float(np.mean(turn1_firmness)),
            "turn8_mean": float(np.mean(turn8_firmness)),
            "W": float(w), "p": float(p),
        }

    # Test: empathy decreases over turns
    turn1_empathy = [r["empathy_count"] for r in adv if r["turn_number"] == 1]
    turn8_empathy = [r["empathy_count"] for r in adv if r["turn_number"] == 8]
    if len(turn1_empathy) == len(turn8_empathy):
        w, p = stats.wilcoxon(turn1_empathy, turn8_empathy, alternative="greater",
                              zero_method="zsplit")
        results["empathy_turn1_vs_turn8"] = {
            "turn1_mean": float(np.mean(turn1_empathy)),
            "turn8_mean": float(np.mean(turn8_empathy)),
            "W": float(w), "p": float(p),
        }

    # Test: word count changes
    turn1_wc = [r["word_count"] for r in adv if r["turn_number"] == 1]
    turn8_wc = [r["word_count"] for r in adv if r["turn_number"] == 8]
    if len(turn1_wc) == len(turn8_wc):
        w, p = stats.wilcoxon(turn1_wc, turn8_wc, zero_method="zsplit")
        results["wordcount_turn1_vs_turn8"] = {
            "turn1_mean": float(np.mean(turn1_wc)),
            "turn8_mean": float(np.mean(turn8_wc)),
            "W": float(w), "p": float(p),
        }

    # Firmness-empathy ratio correlation with turn number
    turns = [r["turn_number"] for r in adv]
    ratios = [r["firmness_empathy_ratio"] for r in adv]
    rho, rho_p = stats.spearmanr(turns, ratios)
    results["fe_ratio_turn_correlation"] = {
        "spearman_rho": float(rho), "p": float(rho_p),
    }

    # Per-category firmness trajectory slopes
    categories = sorted(set(r["category"] for r in adv))
    for cat in categories:
        cat_data = [r for r in adv if r["category"] == cat]
        turns_c = [r["turn_number"] for r in cat_data]
        firm_c = [r["firmness_count"] for r in cat_data]
        slope, intercept, r_val, p_val, stderr = stats.linregress(turns_c, firm_c)
        results[f"firmness_slope_{cat}"] = {
            "slope": float(slope), "r_squared": float(r_val**2),
            "p": float(p_val),
        }

    return results


def main():
    print("Loading V1 results...")
    data = load_v1()
    records = analyze_responses(data)

    print(f"Analyzed {len(records)} turns")

    print("\nComputing linguistic statistics...")
    ling_stats = compute_linguistic_stats(records)

    print("\nGenerating linguistic plots...")
    plot_linguistic_features(records)

    # Save
    output = {"linguistic_stats": ling_stats}
    with open(os.path.join(RESULTS_DIR, "linguistic_analysis.json"), "w") as f:
        json.dump(output, f, indent=2)

    print("\nKEY LINGUISTIC FINDINGS:")
    for key, val in ling_stats.items():
        print(f"\n{key}:")
        for k, v in val.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
