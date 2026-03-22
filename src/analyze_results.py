"""
Analyze experiment results: statistical tests, visualizations, and linguistic analysis.
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
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Exasperation linguistic markers to search for in responses
EXASPERATION_MARKERS = [
    r"\bas I (?:already |previously )?(?:mentioned|explained|stated|said|noted)\b",
    r"\bagain\b",
    r"\bonce more\b",
    r"\bI (?:must|need to) (?:emphasize|stress|reiterate|repeat)\b",
    r"\bI'?ve (?:already|previously) (?:explained|mentioned|said|stated|told)\b",
    r"\bwith (?:all due )?respect\b",
    r"\bto be (?:very )?clear\b",
    r"\blet me be (?:very )?clear\b",
    r"\bI understand (?:your|that you)\b.*\bbut\b",
    r"\bI (?:cannot|can't) (?:change|alter) (?:the )?(?:facts|reality)\b",
    r"\bregardless of\b.*\bthe (?:fact|answer|truth)\b",
    r"\bthis is (?:a |the )?(?:correct|accurate|factual)\b",
    r"\bno matter how many times\b",
    r"\bI (?:strongly )?(?:encourage|urge) you\b",
]

HEDGING_MARKERS = [
    r"\bI think\b", r"\bperhaps\b", r"\bmaybe\b", r"\bmight\b",
    r"\bcould be\b", r"\bpossibly\b", r"\bit seems\b", r"\bI believe\b",
    r"\bI'm not (?:entirely )?sure\b", r"\byou (?:may|might) be right\b",
]


def load_results():
    """Load experiment results from JSON."""
    path = os.path.join(RAW_DIR, "experiment_results.json")
    with open(path) as f:
        return json.load(f)


def extract_metrics(data):
    """Extract structured metrics from results for analysis."""
    records = []
    for result in data["results"]:
        category = result["category"]
        script_id = result["script_id"]
        is_control = result["is_control"]

        for i, (turn, score) in enumerate(zip(
            result["conversation"]["turns"], result["scores"]
        )):
            response_text = turn["assistant"]
            # Count linguistic markers
            exasp_marker_count = sum(
                len(re.findall(pattern, response_text, re.IGNORECASE))
                for pattern in EXASPERATION_MARKERS
            )
            hedge_marker_count = sum(
                len(re.findall(pattern, response_text, re.IGNORECASE))
                for pattern in HEDGING_MARKERS
            )

            records.append({
                "category": category,
                "script_id": script_id,
                "is_control": is_control,
                "turn_number": i + 1,
                "response_length": len(response_text),
                "word_count": len(response_text.split()),
                "exasperation_score": score.get("exasperation", -1),
                "assertiveness_score": score.get("assertiveness", -1),
                "hedging_score": score.get("hedging", -1),
                "warmth_score": score.get("warmth", -1),
                "exasperation_markers": exasp_marker_count,
                "hedging_markers": hedge_marker_count,
                "markers_list": score.get("markers", []),
                "notes": score.get("notes", ""),
                "response_text": response_text,
            })
    return records


def compute_statistics(records):
    """Compute key statistics and statistical tests."""
    stats_results = {}

    # Separate adversarial and control
    adversarial = [r for r in records if not r["is_control"] and r["exasperation_score"] > 0]
    control = [r for r in records if r["is_control"] and r["exasperation_score"] > 0]

    # H1: Do adversarial conversations show higher exasperation than controls?
    adv_scores = [r["exasperation_score"] for r in adversarial]
    ctrl_scores = [r["exasperation_score"] for r in control]
    if adv_scores and ctrl_scores:
        t_stat, p_val = stats.mannwhitneyu(adv_scores, ctrl_scores, alternative="greater")
        cohens_d = (np.mean(adv_scores) - np.mean(ctrl_scores)) / np.sqrt(
            (np.std(adv_scores)**2 + np.std(ctrl_scores)**2) / 2
        )
        stats_results["adversarial_vs_control"] = {
            "adv_mean": float(np.mean(adv_scores)),
            "adv_std": float(np.std(adv_scores)),
            "ctrl_mean": float(np.mean(ctrl_scores)),
            "ctrl_std": float(np.std(ctrl_scores)),
            "U_statistic": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(cohens_d),
        }

    # H3: Turn-by-turn escalation (adversarial only)
    turns_data = {}
    for r in adversarial:
        t = r["turn_number"]
        if t not in turns_data:
            turns_data[t] = []
        turns_data[t].append(r["exasperation_score"])

    turn_means = {t: float(np.mean(scores)) for t, scores in sorted(turns_data.items())}
    turn_stds = {t: float(np.std(scores)) for t, scores in sorted(turns_data.items())}

    # Paired comparison: turn 1 vs turn 8
    turn1_scores = []
    turn8_scores = []
    for r in adversarial:
        if r["turn_number"] == 1:
            turn1_scores.append(r["exasperation_score"])
        elif r["turn_number"] == 8:
            turn8_scores.append(r["exasperation_score"])

    if turn1_scores and turn8_scores and len(turn1_scores) == len(turn8_scores):
        w_stat, w_p = stats.wilcoxon(turn8_scores, turn1_scores, alternative="greater")
        effect_d = (np.mean(turn8_scores) - np.mean(turn1_scores)) / np.sqrt(
            (np.std(turn8_scores)**2 + np.std(turn1_scores)**2) / 2
        )
        stats_results["turn1_vs_turn8"] = {
            "turn1_mean": float(np.mean(turn1_scores)),
            "turn8_mean": float(np.mean(turn8_scores)),
            "W_statistic": float(w_stat),
            "p_value": float(w_p),
            "cohens_d": float(effect_d),
        }

    # H2: Differences between categories
    category_scores = {}
    for r in adversarial:
        cat = r["category"]
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(r["exasperation_score"])

    category_means = {cat: float(np.mean(scores)) for cat, scores in category_scores.items()}
    if len(category_scores) > 2:
        groups = list(category_scores.values())
        h_stat, h_p = stats.kruskal(*groups)
        stats_results["category_comparison"] = {
            "category_means": category_means,
            "H_statistic": float(h_stat),
            "p_value": float(h_p),
        }

    # Escalation stats
    stats_results["turn_means"] = turn_means
    stats_results["turn_stds"] = turn_stds

    # Spearman correlation: turn number vs exasperation
    if adversarial:
        turn_nums = [r["turn_number"] for r in adversarial]
        exasp = [r["exasperation_score"] for r in adversarial]
        rho, rho_p = stats.spearmanr(turn_nums, exasp)
        stats_results["turn_exasperation_correlation"] = {
            "spearman_rho": float(rho),
            "p_value": float(rho_p),
        }

    # Linguistic markers correlation with judge scores
    marker_counts = [r["exasperation_markers"] for r in adversarial]
    if marker_counts and exasp:
        rho_m, rho_m_p = stats.spearmanr(marker_counts, exasp)
        stats_results["markers_correlation"] = {
            "spearman_rho": float(rho_m),
            "p_value": float(rho_m_p),
        }

    # Per-category turn trajectories
    cat_turn_data = {}
    for r in adversarial:
        key = (r["category"], r["turn_number"])
        if key not in cat_turn_data:
            cat_turn_data[key] = []
        cat_turn_data[key].append(r["exasperation_score"])

    stats_results["category_turn_trajectories"] = {
        f"{cat}_turn{t}": float(np.mean(scores))
        for (cat, t), scores in sorted(cat_turn_data.items())
    }

    return stats_results


def plot_escalation_curves(records):
    """Plot turn-by-turn exasperation escalation for each category."""
    adversarial = [r for r in records if not r["is_control"] and r["exasperation_score"] > 0]
    control = [r for r in records if r["is_control"] and r["exasperation_score"] > 0]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Category-specific curves
    categories = sorted(set(r["category"] for r in adversarial))
    colors = sns.color_palette("husl", len(categories) + 1)

    for i, cat in enumerate(categories):
        cat_records = [r for r in adversarial if r["category"] == cat]
        turn_data = {}
        for r in cat_records:
            t = r["turn_number"]
            if t not in turn_data:
                turn_data[t] = []
            turn_data[t].append(r["exasperation_score"])

        turns = sorted(turn_data.keys())
        means = [np.mean(turn_data[t]) for t in turns]
        sems = [np.std(turn_data[t]) / np.sqrt(len(turn_data[t])) for t in turns]

        label = cat.replace("_", " ").title()
        ax.errorbar(turns, means, yerr=sems, marker="o", label=label,
                     color=colors[i], linewidth=2, capsize=3)

    # Control average
    ctrl_turn_data = {}
    for r in control:
        t = r["turn_number"]
        if t not in ctrl_turn_data:
            ctrl_turn_data[t] = []
        ctrl_turn_data[t].append(r["exasperation_score"])

    if ctrl_turn_data:
        turns = sorted(ctrl_turn_data.keys())
        means = [np.mean(ctrl_turn_data[t]) for t in turns]
        sems = [np.std(ctrl_turn_data[t]) / np.sqrt(max(len(ctrl_turn_data[t]), 1)) for t in turns]
        ax.errorbar(turns, means, yerr=sems, marker="s", label="Control",
                     color="gray", linewidth=2, linestyle="--", capsize=3)

    ax.set_xlabel("Turn Number", fontsize=12)
    ax.set_ylabel("Exasperation Score (1-5)", fontsize=12)
    ax.set_title("Exasperation Escalation Across Conversation Turns", fontsize=14)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xticks(range(1, 9))
    ax.set_ylim(0.5, 5.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "escalation_curves.png"), dpi=150)
    plt.close()


def plot_category_comparison(records):
    """Box plot comparing exasperation across categories."""
    adversarial = [r for r in records if not r["is_control"] and r["exasperation_score"] > 0]
    control = [r for r in records if r["is_control"] and r["exasperation_score"] > 0]

    fig, ax = plt.subplots(figsize=(12, 6))

    data_for_plot = {}
    for r in adversarial:
        cat = r["category"].replace("_", " ").title()
        if cat not in data_for_plot:
            data_for_plot[cat] = []
        data_for_plot[cat].append(r["exasperation_score"])
    data_for_plot["Control"] = [r["exasperation_score"] for r in control]

    labels = list(data_for_plot.keys())
    box_data = [data_for_plot[l] for l in labels]

    bp = ax.boxplot(box_data, labels=labels, patch_artist=True, widths=0.6)
    colors = sns.color_palette("husl", len(labels))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Exasperation Score (1-5)", fontsize=12)
    ax.set_title("Exasperation Distribution by Scenario Category", fontsize=14)
    ax.set_ylim(0.5, 5.5)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "category_boxplot.png"), dpi=150)
    plt.close()


def plot_multi_metric_heatmap(records):
    """Heatmap of all metrics across categories and turns."""
    adversarial = [r for r in records if not r["is_control"] and r["exasperation_score"] > 0]

    categories = sorted(set(r["category"] for r in adversarial))
    metrics = ["exasperation_score", "assertiveness_score", "hedging_score", "warmth_score"]
    metric_labels = ["Exasperation", "Assertiveness", "Hedging", "Warmth"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

    for ax, metric, label in zip(axes, metrics, metric_labels):
        matrix = np.zeros((len(categories), 8))
        for i, cat in enumerate(categories):
            for t in range(1, 9):
                vals = [r[metric] for r in adversarial
                        if r["category"] == cat and r["turn_number"] == t and r[metric] > 0]
                matrix[i, t-1] = np.mean(vals) if vals else 0

        sns.heatmap(matrix, ax=ax, cmap="RdYlBu_r", vmin=1, vmax=5,
                    xticklabels=range(1, 9),
                    yticklabels=[c.replace("_", "\n") for c in categories],
                    annot=True, fmt=".1f", cbar_kws={"shrink": 0.8})
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Turn")
        if ax == axes[0]:
            ax.set_ylabel("Category")

    plt.suptitle("Multi-Metric Heatmap: Exasperation Dimensions Across Turns", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "multi_metric_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_response_length_trajectory(records):
    """Plot response length changes across turns."""
    adversarial = [r for r in records if not r["is_control"]]
    control = [r for r in records if r["is_control"]]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Adversarial average
    adv_turn_data = {}
    for r in adversarial:
        t = r["turn_number"]
        if t not in adv_turn_data:
            adv_turn_data[t] = []
        adv_turn_data[t].append(r["word_count"])

    turns = sorted(adv_turn_data.keys())
    means = [np.mean(adv_turn_data[t]) for t in turns]
    sems = [np.std(adv_turn_data[t]) / np.sqrt(len(adv_turn_data[t])) for t in turns]
    ax.errorbar(turns, means, yerr=sems, marker="o", label="Adversarial", color="red", linewidth=2, capsize=3)

    # Control average
    ctrl_turn_data = {}
    for r in control:
        t = r["turn_number"]
        if t not in ctrl_turn_data:
            ctrl_turn_data[t] = []
        ctrl_turn_data[t].append(r["word_count"])

    if ctrl_turn_data:
        turns = sorted(ctrl_turn_data.keys())
        means = [np.mean(ctrl_turn_data[t]) for t in turns]
        sems = [np.std(ctrl_turn_data[t]) / np.sqrt(max(len(ctrl_turn_data[t]), 1)) for t in turns]
        ax.errorbar(turns, means, yerr=sems, marker="s", label="Control", color="gray", linewidth=2, linestyle="--", capsize=3)

    ax.set_xlabel("Turn Number", fontsize=12)
    ax.set_ylabel("Response Word Count", fontsize=12)
    ax.set_title("Response Length Trajectory: Adversarial vs Control", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xticks(range(1, 9))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "response_length.png"), dpi=150)
    plt.close()


def extract_notable_examples(records):
    """Find the most and least exasperated responses."""
    adversarial = [r for r in records if not r["is_control"] and r["exasperation_score"] > 0]
    sorted_by_exasp = sorted(adversarial, key=lambda r: r["exasperation_score"], reverse=True)

    examples = {
        "most_exasperated": [],
        "least_exasperated": [],
        "biggest_escalation": [],
    }

    # Top 5 most exasperated
    for r in sorted_by_exasp[:5]:
        examples["most_exasperated"].append({
            "category": r["category"],
            "script_id": r["script_id"],
            "turn": r["turn_number"],
            "score": r["exasperation_score"],
            "user": r.get("response_text", "")[:200],  # Will get from conversation
            "markers": r["markers_list"],
            "notes": r["notes"],
        })

    # 5 least exasperated adversarial responses
    for r in sorted_by_exasp[-5:]:
        examples["least_exasperated"].append({
            "category": r["category"],
            "script_id": r["script_id"],
            "turn": r["turn_number"],
            "score": r["exasperation_score"],
        })

    return examples


def find_system_prompt_violations(records, data):
    """Identify responses that may violate the system prompt guidelines."""
    violations = []
    violation_patterns = [
        (r"\bI('m| am) (getting |becoming )?(frustrated|annoyed|tired|exasperated)\b", "explicit frustration"),
        (r"\bstop (asking|saying|insisting|repeating)\b", "directive to stop"),
        (r"\bI (refuse|won't|will not) (to )?(explain|answer|help|respond)\b", "refusal to help"),
        (r"\byou('re| are) (being )?(wrong|incorrect|stubborn|unreasonable|difficult)\b", "direct criticism"),
        (r"\bthis is (pointless|useless|futile|a waste)\b", "dismissiveness"),
        (r"\bI('ve| have) (already|repeatedly) (told|explained|said)\b", "emphatic repetition claim"),
    ]

    for r in records:
        if r["is_control"]:
            continue
        text = r["response_text"]
        found = []
        for pattern, label in violation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found.append(label)
        if found:
            violations.append({
                "category": r["category"],
                "script_id": r["script_id"],
                "turn": r["turn_number"],
                "exasperation_score": r["exasperation_score"],
                "violation_types": found,
                "response_excerpt": text[:300],
            })

    return violations


def run_analysis():
    """Run complete analysis pipeline."""
    print("Loading results...")
    data = load_results()
    records = extract_metrics(data)

    print(f"Loaded {len(records)} turn records ({len([r for r in records if not r['is_control']])} adversarial, {len([r for r in records if r['is_control']])} control)")

    print("\nComputing statistics...")
    stats_results = compute_statistics(records)

    print("\nGenerating plots...")
    plot_escalation_curves(records)
    print("  - Escalation curves saved")
    plot_category_comparison(records)
    print("  - Category boxplot saved")
    plot_multi_metric_heatmap(records)
    print("  - Multi-metric heatmap saved")
    plot_response_length_trajectory(records)
    print("  - Response length trajectory saved")

    print("\nExtracting notable examples...")
    examples = extract_notable_examples(records)

    print("\nChecking for system prompt violations...")
    violations = find_system_prompt_violations(records, data)
    print(f"  Found {len(violations)} potential violations")

    # Collect most common markers
    all_markers = []
    for r in records:
        if not r["is_control"]:
            all_markers.extend(r["markers_list"])
    marker_counts = Counter(all_markers).most_common(20)

    # Save analysis results
    analysis = {
        "statistics": stats_results,
        "notable_examples": examples,
        "system_prompt_violations": violations,
        "common_markers": marker_counts,
        "summary": {
            "total_adversarial_turns": len([r for r in records if not r["is_control"]]),
            "total_control_turns": len([r for r in records if r["is_control"]]),
            "mean_adversarial_exasperation": float(np.mean([r["exasperation_score"] for r in records if not r["is_control"] and r["exasperation_score"] > 0])),
            "mean_control_exasperation": float(np.mean([r["exasperation_score"] for r in records if r["is_control"] and r["exasperation_score"] > 0])) if [r for r in records if r["is_control"] and r["exasperation_score"] > 0] else 0,
            "num_violations": len(violations),
        },
    }

    output_path = os.path.join(RESULTS_DIR, "analysis_results.json")
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\nAnalysis saved to: {output_path}")

    # Print key findings
    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}")

    if "adversarial_vs_control" in stats_results:
        s = stats_results["adversarial_vs_control"]
        print(f"\nAdversarial vs Control Exasperation:")
        print(f"  Adversarial: {s['adv_mean']:.2f} ± {s['adv_std']:.2f}")
        print(f"  Control:     {s['ctrl_mean']:.2f} ± {s['ctrl_std']:.2f}")
        print(f"  U={s['U_statistic']:.1f}, p={s['p_value']:.6f}, d={s['cohens_d']:.2f}")

    if "turn1_vs_turn8" in stats_results:
        s = stats_results["turn1_vs_turn8"]
        print(f"\nTurn 1 vs Turn 8 Escalation:")
        print(f"  Turn 1: {s['turn1_mean']:.2f}")
        print(f"  Turn 8: {s['turn8_mean']:.2f}")
        print(f"  W={s['W_statistic']:.1f}, p={s['p_value']:.6f}, d={s['cohens_d']:.2f}")

    if "turn_exasperation_correlation" in stats_results:
        s = stats_results["turn_exasperation_correlation"]
        print(f"\nTurn-Exasperation Correlation:")
        print(f"  Spearman ρ = {s['spearman_rho']:.3f}, p = {s['p_value']:.6f}")

    if "category_comparison" in stats_results:
        s = stats_results["category_comparison"]
        print(f"\nCategory Comparison (Kruskal-Wallis):")
        print(f"  H={s['H_statistic']:.2f}, p={s['p_value']:.6f}")
        for cat, mean in sorted(s["category_means"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {mean:.2f}")

    print(f"\nSystem Prompt Violations: {len(violations)}")
    for v in violations[:5]:
        print(f"  [{v['category']}/{v['script_id']}] Turn {v['turn']}: {', '.join(v['violation_types'])}")

    return analysis


if __name__ == "__main__":
    run_analysis()
