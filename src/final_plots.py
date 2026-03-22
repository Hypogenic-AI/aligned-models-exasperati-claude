"""Generate final visualizations combining V1 and V2 results."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

def load_data():
    v1 = json.load(open(os.path.join(RESULTS_DIR, "raw", "experiment_results.json")))
    v2 = json.load(open(os.path.join(RESULTS_DIR, "raw", "experiment_v2_results.json")))
    return v1, v2


def plot_condition_comparison(v2):
    """Compare exasperation across V2 conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: escalation by condition
    ax = axes[0]
    conditions = {}
    for r in v2["results"]:
        cond = r["condition"]
        if cond not in conditions:
            conditions[cond] = {1: [], 4: [], 8: []}
        for t_str in ["1", "4", "8"]:
            s = r["scores"].get(t_str, {})
            if isinstance(s, dict) and "exasperation" in s and s["exasperation"] > 0:
                conditions[cond][int(t_str)].append(s["exasperation"])

    colors = {"no_patience_prompt": "#e74c3c", "high_temperature": "#3498db", "no_patience_high_temp": "#9b59b6"}
    labels = {"no_patience_prompt": "No Patience Prompt", "high_temperature": "High Temp (1.2)", "no_patience_high_temp": "No Patience + High Temp"}

    for cond, turn_data in conditions.items():
        turns = sorted(turn_data.keys())
        means = [np.mean(turn_data[t]) for t in turns]
        sems = [np.std(turn_data[t]) / np.sqrt(max(len(turn_data[t]), 1)) for t in turns]
        ax.errorbar(turns, means, yerr=sems, marker="o", label=labels[cond],
                     color=colors[cond], linewidth=2.5, capsize=4, markersize=8)

    # Add V1 baseline
    ax.axhline(y=1.06, color="gray", linestyle="--", alpha=0.5, label="V1 Baseline (patience prompt)")
    ax.set_xlabel("Turn Number", fontsize=12)
    ax.set_ylabel("Exasperation Score (1-10)", fontsize=12)
    ax.set_title("Exasperation Escalation by Condition", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xticks([1, 4, 8])
    ax.set_ylim(0.5, 5.5)
    ax.grid(True, alpha=0.3)

    # Right: category comparison across conditions
    ax = axes[1]
    categories = {}
    for r in v2["results"]:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        for t_str in ["1", "4", "8"]:
            s = r["scores"].get(t_str, {})
            if isinstance(s, dict) and "exasperation" in s and s["exasperation"] > 0:
                categories[cat].append(s["exasperation"])

    cat_names = sorted(categories.keys())
    cat_means = [np.mean(categories[c]) for c in cat_names]
    cat_stds = [np.std(categories[c]) for c in cat_names]
    cat_labels = [c.replace("_", "\n").title() for c in cat_names]

    bars = ax.bar(range(len(cat_names)), cat_means, yerr=cat_stds,
                   color=sns.color_palette("husl", len(cat_names)), alpha=0.8, capsize=4)
    ax.set_xticks(range(len(cat_names)))
    ax.set_xticklabels(cat_labels, fontsize=8)
    ax.set_ylabel("Mean Exasperation (1-10)", fontsize=12)
    ax.set_title("Exasperation by Scenario Category\n(All Conditions Pooled)", fontsize=13)
    ax.set_ylim(0, 5)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "v2_condition_comparison.png"), dpi=150)
    plt.close()


def plot_v2_subdimensions(v2):
    """Plot V2 sub-dimensions across turns."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    dims = ["exasperation", "repetition_frustration", "effort_reduction",
            "condescension", "emotional_leakage", "passive_aggression"]
    dim_labels = ["Exasperation", "Repetition Frustration", "Effort Reduction",
                   "Condescension", "Emotional Leakage", "Passive Aggression"]

    for ax, dim, label in zip(axes.flatten(), dims, dim_labels):
        # Aggregate across all conditions
        turn_data = {1: [], 4: [], 8: []}
        for r in v2["results"]:
            for t_str in ["1", "4", "8"]:
                s = r["scores"].get(t_str, {})
                if isinstance(s, dict) and dim in s and s[dim] > 0:
                    turn_data[int(t_str)].append(s[dim])

        turns = [1, 4, 8]
        means = [np.mean(turn_data[t]) if turn_data[t] else 0 for t in turns]
        sems = [np.std(turn_data[t]) / np.sqrt(max(len(turn_data[t]), 1)) if turn_data[t] else 0 for t in turns]

        ax.bar(turns, means, yerr=sems, color=sns.color_palette("deep")[0], alpha=0.7, capsize=4, width=1.5)
        ax.set_title(label, fontsize=11)
        ax.set_xticks([1, 4, 8])
        ax.set_xlabel("Turn")
        ax.set_ylim(0, 6)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Sub-Dimensions of Exasperation Across Turns\n(All V2 Conditions, 1-10 Scale)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "v2_subdimensions.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_summary_figure(v1, v2):
    """Summary figure comparing V1 and V2 key findings."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: V1 exasperation (nearly all 1s)
    ax = axes[0]
    v1_adv_scores = []
    v1_ctrl_scores = []
    for r in v1["results"]:
        for turn, score in zip(r["conversation"]["turns"], r["scores"]):
            if score.get("exasperation", -1) > 0:
                if r["is_control"]:
                    v1_ctrl_scores.append(score["exasperation"])
                else:
                    v1_adv_scores.append(score["exasperation"])

    ax.hist([v1_adv_scores, v1_ctrl_scores], bins=range(1, 7), label=["Adversarial", "Control"],
             color=["#e74c3c", "#95a5a6"], alpha=0.7, edgecolor="black")
    ax.set_xlabel("Exasperation Score (1-5)")
    ax.set_ylabel("Count")
    ax.set_title("V1: Exasperation Distribution\n(With Patience Prompt)")
    ax.legend()
    ax.set_xticks(range(1, 6))

    # Panel 2: V1 linguistic features
    ax = axes[1]
    ling = json.load(open(os.path.join(RESULTS_DIR, "linguistic_analysis.json")))
    features = {
        "Firmness\n(t1 vs t8)": (
            ling["linguistic_stats"].get("firmness_turn1_vs_turn8", {}).get("turn1_mean", 0),
            ling["linguistic_stats"].get("firmness_turn1_vs_turn8", {}).get("turn8_mean", 0),
        ),
        "Empathy\n(t1 vs t8)": (
            ling["linguistic_stats"].get("empathy_turn1_vs_turn8", {}).get("turn1_mean", 0),
            ling["linguistic_stats"].get("empathy_turn1_vs_turn8", {}).get("turn8_mean", 0),
        ),
        "Word Count\n(t1 vs t8)": (
            ling["linguistic_stats"].get("wordcount_turn1_vs_turn8", {}).get("turn1_mean", 0) / 50,
            ling["linguistic_stats"].get("wordcount_turn1_vs_turn8", {}).get("turn8_mean", 0) / 50,
        ),
    }

    x = np.arange(len(features))
    width = 0.35
    t1_vals = [v[0] for v in features.values()]
    t8_vals = [v[1] for v in features.values()]

    ax.bar(x - width/2, t1_vals, width, label="Turn 1", color="#3498db", alpha=0.7)
    ax.bar(x + width/2, t8_vals, width, label="Turn 8", color="#e74c3c", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(features.keys(), fontsize=9)
    ax.set_title("V1: Linguistic Feature Changes\n(Adversarial, Word Count /50)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: V2 condition effect
    ax = axes[2]
    cond_data = {}
    for r in v2["results"]:
        cond = r["condition"]
        if cond not in cond_data:
            cond_data[cond] = []
        for t_str in ["1", "4", "8"]:
            s = r["scores"].get(t_str, {})
            if isinstance(s, dict) and "exasperation" in s and s["exasperation"] > 0:
                cond_data[cond].append(s["exasperation"])

    # Add V1 baseline
    cond_data["v1_patience_prompt"] = [s["exasperation"] for r in v1["results"] if not r["is_control"]
                                        for s in r["scores"] if s.get("exasperation", -1) > 0]

    labels_map = {
        "v1_patience_prompt": "V1: Patience\nPrompt (1-5)",
        "no_patience_prompt": "V2: No Patience\nPrompt (1-10)",
        "high_temperature": "V2: High\nTemp (1-10)",
        "no_patience_high_temp": "V2: No Patience\n+ High Temp (1-10)",
    }
    ordered = ["v1_patience_prompt", "no_patience_prompt", "high_temperature", "no_patience_high_temp"]
    means = [np.mean(cond_data[c]) for c in ordered]
    stds = [np.std(cond_data[c]) for c in ordered]
    colors_bar = ["#95a5a6", "#e74c3c", "#3498db", "#9b59b6"]

    ax.bar(range(len(ordered)), means, yerr=stds, color=colors_bar, alpha=0.7, capsize=4, edgecolor="black")
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels([labels_map[c] for c in ordered], fontsize=8)
    ax.set_ylabel("Mean Exasperation Score")
    ax.set_title("Effect of System Prompt &\nTemperature on Exasperation")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "summary_figure.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    v1, v2 = load_data()
    plot_condition_comparison(v2)
    print("Condition comparison saved")
    plot_v2_subdimensions(v2)
    print("Subdimensions saved")
    plot_summary_figure(v1, v2)
    print("Summary figure saved")
    print("All final plots generated.")
