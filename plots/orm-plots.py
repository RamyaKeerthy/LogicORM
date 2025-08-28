from __future__ import annotations

import os
import re
import glob
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

try:
    import seaborn as sns  # optional; used only for a pastel palette
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False


# ============================== Config ==============================

BASE_DIR = "path/to/ORM/output"

# Dataset sizes used to take the last N rows from each file
DATASET_SIZE = {"folio": 203, "proverqa": 500, "justlogic": 600}

# File glob patterns (cot vs echo) per dataset
PATTERNS: Dict[str, Tuple[str, str]] = {
    "justlogic": (
        os.path.join(BASE_DIR, "justlogic_*_output_sample_32_orm_cot-gpt4o-sample8-resampled10k.jsonl"),
        os.path.join(BASE_DIR, "justlogic_*_output_sample_32_orm_echo-gpt4o-sample8-resampled10k.jsonl"),
    ),
    "folio": (
        os.path.join(BASE_DIR, "folio_*_output_sample_32_orm_cot-gpt4o-sample10.jsonl"),
        os.path.join(BASE_DIR, "folio_*_output_sample_32_orm_echo-gpt4o-sample10.jsonl"),
    ),
    "proverqa": (
        os.path.join(BASE_DIR, "proverqa_*_output_sample_32_orm_cot-gpt4o-sample8.jsonl"),
        os.path.join(BASE_DIR, "proverqa_*_output_sample_32_orm_echo-gpt4o-sample8.jsonl"),
    ),
}

# Batch sizes (sample counts) to evaluate
BATCH_SIZES = [1, 4, 8, 16, 32]

# Titles for the main comparison grid (left as non-personal model names)
MODEL_TITLES = {
    "qwen25": "Qwen2.5-7B-Instruct Reasoner",
    "gpt4o": "GPT4o Reasoner",
    "llama31": "Llama-3.1-8B-Instruct Reasoner",
    "qwen3": "Qwen3-8B Reasoner",
    "gemma12": "Gemma3-12B Reasoner",
    "gemma1": "Gemma3-1B Reasoner",
    "gemma4": "Gemma3-4B Reasoner",
}

# Palette
COLORS = {
    "Majority": "#4C72B0",
    "ORM-Cot": "#55A868",
    "ORM-echo": "#C44E52",
    "HT": "red"}


# ======================== Normalization Helpers ========================

_A_VARIANTS = {
    "A", "A) TRUE", "TRUE", "TRUE (A)", "CORRECT", "YES", "A).", "A) \\", "A) \\ TRUE",
    "\\TEXT{A IS TRUE 1", "A = \\TEXT{TRUE", "\\TEXT{TRUE", "A \\TEXT{) TRUE", "A) \\TEXT{TRUE",
    "A)", "\\TEXT{A)", "\\TEXT{A", "\\TEXT{A) TRUE", "A). TRUE", "A). TRUE.",
}
_B_VARIANTS = {
    "B", "B) FALSE", "FALSE", "FALSE (B)", "INCORRECT", "NO", "B)", "B) \\TEXT{FALSE",
    "\\TEXT{B)", "\\TEXT{FALSE", "\\TEXT{B) FALSE", "B \\TEXT{ (FALSE)",
}
_C_VARIANTS = {
    "C", "C) UNCERTAIN", "UNKNOWN", "INDETERMINATE", "CANNOT BE DETERMINED",
    "UNCERTAIN", "UNCERTAIN (C)", "C: UNCERTAIN",
    "C) \\TEXT{UNCERTAIN", "C)", "C \\TEXT{) UNCERTAIN", "C \\TEXT{) UNCERTAIN 1",
    "\\TEXT{C)", "\\TEXT{C", "\\TEXT{C) UNCERTAIN", "\\TEXT{UNCERTAIN", "\\TEXT{uncertain",
}

def _norm(s: Union[str, bool]) -> str:
    if isinstance(s, bool):
        return "TRUE" if s else "FALSE"
    return re.sub(r"\s+", " ", str(s).strip()).upper()

def correct_options(answer: Union[str, bool]) -> Union[str, bool]:
    """
    Map many textual variants to canonical {A,B,C}. If no mapping, return original.
    """
    token = _norm(answer)
    if token in _A_VARIANTS or token.startswith("A"):
        return "A"
    if token in _B_VARIANTS or token.startswith("B"):
        return "B"
    if token in _C_VARIANTS or token.startswith("C") or token in {"UNCERTAIN", "UNKNOWN"}:
        return "C"
    return answer


# =========================== Extraction Utils ===========================

_BOXED = re.compile(r"\\boxed\{(.*?)\}")

def extract_boxed_answers(text_or_list: Union[str, List[str]]) -> List[str]:
    """
    Extract content inside \boxed{...}. Accepts a string or list[str].
    Returns list[str] (empty strings for misses to preserve lengths).
    """
    if isinstance(text_or_list, str):
        texts = [text_or_list]
    else:
        texts = list(text_or_list)

    out: List[str] = []
    for t in texts:
        if not isinstance(t, str):
            out.append("")
            continue
        m = _BOXED.findall(t)
        out.append(m[0] if m else "")
    return out


# ============================ Scoring Functions ============================

def highest_threshold_accuracy(df: pd.DataFrame, prediction_col: str = "predictions", gold_col: str = "gold") -> float:
    """
    Highest-Threshold Accuracy:
    Row is correct if ANY predicted option in prediction_col matches the gold label.
    """
    preds = df[prediction_col].apply(lambda xs: [correct_options(x) for x in xs])
    correct = (df[gold_col].astype(str).str.upper().str[0] == preds.apply(lambda xs: next((x for x in xs if x in {"A","B","C"}), "Z"))).tolist()

    # The above ‘next’ isn’t quite HT; compute properly:
    correct = df.apply(lambda r: r[gold_col] in preds.loc[r.name], axis=1)
    return float(correct.mean() * 100.0)

def get_majority_vote(orm: Sequence, predicted_answer: Sequence, batch: int = 1) -> str:
    """
    Majority vote over the first `batch` items of predicted_answer after normalization.
    `orm` is unused for voting but kept for signature parity / future weighting.
    """
    if not isinstance(predicted_answer, list) or len(predicted_answer) == 0:
        return ""
    votes = [correct_options(a) for a in predicted_answer[:batch]]
    if not votes:
        return ""
    majority_value, _ = Counter(votes).most_common(1)[0]
    return majority_value

def pick_best_value(orm: Sequence, value: Sequence, batch: int = 64) -> str:
    """
    Select value with highest ORM rank within first `batch` items.
    If ties, break by majority vote among tied values.
    """
    if not isinstance(orm, list) or not isinstance(value, list) or not orm:
        return ""
    ranks = orm[:batch]
    vals = value[:batch]
    if not ranks:
        return ""
    max_rank = max(ranks)
    best_idx = [i for i, r in enumerate(ranks) if r == max_rank]
    if len(best_idx) == 1:
        return correct_options(vals[best_idx[0]])
    tied = [correct_options(vals[i]) for i in best_idx]
    return Counter(tied).most_common(1)[0][0]

def get_majority_vote_frequency(orm: Sequence, predicted_answer: Sequence, batch: int = 1) -> float:
    """
    Frequency (count) of the majority vote within the first `batch` predictions.
    """
    if not isinstance(predicted_answer, list) or not predicted_answer:
        return 0.0
    votes = [correct_options(a) for a in predicted_answer[:batch]]
    if not votes:
        return 0.0
    _, freq = Counter(votes).most_common(1)[0]
    return float(freq)


# I/O & Parsing

def _extract_dataset_name_from_pattern(pattern: str) -> str:
    return os.path.basename(pattern).split("_")[0]

def _extract_file_id(path: str) -> str:
    # filename schema: <dataset>_<file_id>_...
    return os.path.basename(path).split("_")[1]

def _map_by_file_id(files: List[str]) -> Dict[str, str]:
    return {_extract_file_id(f): f for f in files}

def _load_last_n(path: str, n: int) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    if n > 0:
        df = df.iloc[-n:]
    return df.copy()

def _evaluate_pair(
    df_cot: pd.DataFrame,
    df_echo: pd.DataFrame,
    dataset: str,
    file_id: str,
    batch_sizes: List[int],
) -> List[Tuple[str, int, float, float, float, float, str]]:
    """
    Compute metrics for one (cot, echo) pair.
    Returns rows: (file_id, batch, majority_vote, best_of_n_cot, best_of_n_echo, ht_acc, dataset)
    """
    df1 = df_cot.copy()
    df2 = df_echo.copy()

    # Extract boxed answers (lists)
    df1["predicted_answer"] = df1["prediction"].apply(extract_boxed_answers)
    df2["predicted_answer"] = df2["prediction"].apply(extract_boxed_answers)

    # Highest-threshold accuracy on CoT set
    ht_acc = highest_threshold_accuracy(df1, prediction_col="predicted_answer", gold_col="gold_answer")

    rows = []
    for b in batch_sizes:
        vote_col = f"majority_vote_{b}"
        best_col = f"best_of_n_boxed_{b}"

        df1[vote_col] = df1.apply(lambda x: get_majority_vote(orm=x["orm_cot_echo"], predicted_answer=x["predicted_answer"], batch=b), axis=1)
        df1[best_col] = df1.apply(lambda x: pick_best_value(orm=x["orm_cot_echo"], value=x["predicted_answer"], batch=b), axis=1)
        df2[best_col] = df2.apply(lambda x: pick_best_value(orm=x["orm_cot_echo"], value=x["predicted_answer"], batch=b), axis=1)

        vote_acc = float((df1["gold_answer"] == df1[vote_col]).mean()) * 100.0
        best_cot_acc = float((df1["gold_answer"] == df1[best_col]).mean()) * 100.0
        best_echo_acc = float((df2["gold_answer"] == df2[best_col]).mean()) * 100.0

        rows.append((file_id, b, np.round(vote_acc, 2), np.round(best_cot_acc, 2), np.round(best_echo_acc, 2), np.round(ht_acc, 2), dataset))
    return rows

def evaluate_dataset(dataset: str, pattern_cot: str, pattern_echo: str) -> pd.DataFrame:
    """
    Evaluate a dataset by pairing up matching CoT/Echo files and aggregating metrics across batch sizes.
    """
    files_1 = glob.glob(pattern_cot)
    files_2 = glob.glob(pattern_echo)
    if not files_1 or not files_2:
        return pd.DataFrame(columns=["file_id", "sample", "majority_vote", "best_of_n_cot", "best_of_n_echo", "ht_acc", "data"])

    map1 = _map_by_file_id(files_1)
    map2 = _map_by_file_id(files_2)
    common = sorted(set(map1.keys()).intersection(set(map2.keys())))
    n_rows = DATASET_SIZE.get(dataset, 0)

    all_rows: List[Tuple[str, int, float, float, float, float, str]] = []
    for fid in common:
        df1 = _load_last_n(map1[fid], n_rows)
        df2 = _load_last_n(map2[fid], n_rows)
        rows = _evaluate_pair(df1, df2, dataset, fid, BATCH_SIZES)
        all_rows.extend(rows)

    return pd.DataFrame(all_rows, columns=["file_id", "sample", "majority_vote", "best_of_n_cot", "best_of_n_echo", "ht_acc", "data"])


def plot_comparison_grid(
    results_df: pd.DataFrame,
    file_ids: List[str],
    datasets: List[str],
    y_min: int = 0,
    y_max: int = 100):
    """
    Recreates your multi-panel comparison with insets and HT lines.
    """
    fig, axes = plt.subplots(len(datasets), len(file_ids), figsize=(5 * len(file_ids), 4 * len(datasets)), sharex=True, sharey=True)
    axes = np.array(axes).reshape(len(datasets), len(file_ids))

    for r, dataset in enumerate(datasets):
        for c, fid in enumerate(file_ids):
            ax = axes[r, c]
            df = results_df[(results_df["file_id"] == fid) & (results_df["data"] == dataset)].sort_values(by="sample")
            if df.empty:
                ax.set_visible(False)
                continue

            ax.plot(df["sample"], df["majority_vote"], linestyle="-", marker="D", label="Majority", color=COLORS["Majority"])
            ax.plot(df["sample"], df["best_of_n_cot"], linestyle="-", marker="D", label="ORM-CoT", color=COLORS["ORM-Cot"])
            ax.plot(df["sample"], df["best_of_n_echo"], linestyle="-", marker="D", label="ORM-EcCoT", color=COLORS["ORM-echo"])

            ht_acc = df["ht_acc"].iloc[0]
            ax.axhline(y=ht_acc, color=COLORS["HT"], linestyle="--", linewidth=1.5)

            if r == len(datasets) - 1:
                ax.set_xlabel("Number of Samples", fontsize=18)
                ax.set_xticks([1, 4, 8, 16, 32])
            else:
                ax.set_xticks([])

            if c == 0:
                ax.set_ylabel(f"{dataset.capitalize()}\nAccuracy (%)", fontsize=18)

            ax.grid(True, color="lightgrey", linewidth=0.5)
            ax.set_ylim(y_min, y_max)
            ax.tick_params(axis="both", which="major", labelsize=15)

            # Inset mini-bar showing batch=32 summary
            final = df[df["sample"] == 32].iloc[0]
            y_vals = [final["majority_vote"], final["best_of_n_cot"], final["best_of_n_echo"]]
            avg_y = float(np.mean(y_vals))
            inset_y = 0.01 if avg_y > 40 else 0.55

            inset_ax = inset_axes(
                ax, width="45%", height="55%",
                bbox_to_anchor=(0.6, inset_y, 0.35, 0.35),
                bbox_transform=ax.transAxes, loc="lower right"
            )
            x = np.arange(3) * 0.5
            inset_ax.bar(x, y_vals, color=[COLORS["Majority"], COLORS["ORM-Cot"], COLORS["ORM-echo"]], width=0.4)
            inset_ax.set_title("Batch=32", fontsize=14)
            inset_ax.set_xticks(x)
            inset_ax.set_xticklabels([])

            if r == 0:
                ax.set_title(MODEL_TITLES.get(fid, fid), fontsize=18)

            if r == 0 and c == len(file_ids) - 1:
                method_legend = [
                    Line2D([0], [0], color=COLORS["Majority"], lw=2, linestyle="-", marker="D", label=r"$\mathrm{Majority}$"),
                    Line2D([0], [0], color=COLORS["ORM-Cot"], lw=2, linestyle="-", marker="D", label=r"$\mathrm{ORM\text{-}CoT}$"),
                    Line2D([0], [0], color=COLORS["ORM-echo"], lw=2, linestyle="-", marker="D", label=r"$\mathrm{ORM\text{-}EcCoT}$"),
                    Line2D([0], [0], color=COLORS["HT"], linestyle="--", lw=1.5, label=r"$\mathrm{Highest\ Threshold}$"),
                ]
                leg = ax.legend(handles=method_legend, loc="lower left", fontsize=16, frameon=True)
                leg.get_frame().set_facecolor("#f0f0f0")
                leg.get_frame().set_edgecolor("#f0f0f0")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_inference_frequency(
    batch_sizes: List[int],
    patterns: Dict[str, str],
    models_dict: Dict[str, str],
    allowed_file_ids: Optional[Iterable[str]] = None):
    """
    Replicates the 'Inference Plots': Avg Majority Vote Frequency vs Batch Size.
    Only CoT files are needed for this plot.
    """
    fig, axes = plt.subplots(1, len(patterns), figsize=(6 * len(patterns), 5), sharey=True)

    if len(patterns) == 1:
        axes = np.array([axes])

    for i, (data_key, pattern) in enumerate(patterns.items()):
        files = glob.glob(pattern)
        df_by_model: Dict[str, pd.DataFrame] = {}
        for path in files:
            fid = _extract_file_id(path)
            if allowed_file_ids and fid not in allowed_file_ids:
                continue
            if fid not in models_dict:
                continue
            name = models_dict[fid]
            df_by_model[name] = pd.read_json(path, lines=True)

        results: Dict[str, List[float]] = {}
        for model_name, df in df_by_model.items():
            df["predicted_answer"] = df["prediction"].apply(extract_boxed_answers)
            avgs: List[float] = []
            for b in batch_sizes:
                freqs = df.apply(
                    lambda x: get_majority_vote_frequency(
                        orm=x["orm_cot_echo"], predicted_answer=x["predicted_answer"], batch=b
                    ),
                    axis=1,
                )
                avgs.append(float(freqs.mean()))
            results[model_name] = avgs

        ax = axes[i]
        for model_name, avgs in results.items():
            ax.plot(batch_sizes, avgs, marker="o", label=model_name)
        ax.set_title(data_key.capitalize(), fontsize=18)
        ax.set_xlabel("Sample Size", fontsize=18)
        if i == 0:
            ax.set_ylabel("Avg Majority Vote Frequency", fontsize=18)
        ax.set_xticks(batch_sizes)
        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.grid(True)
        if i == 0:
            ax.legend(fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    # Aggregate results across datasets
    per_dataset_frames = []
    for dataset, (pat_cot, pat_echo) in PATTERNS.items():
        df_res = evaluate_dataset(dataset, pat_cot, pat_echo)
        per_dataset_frames.append(df_res)

    results_df = pd.concat(per_dataset_frames, ignore_index=True) if per_dataset_frames else pd.DataFrame()

    # Choose which models & datasets to visualize
    file_ids = ["gemma1", "gemma4", "gemma12"]  # e.g., ["qwen25", "gpt4o", "llama31"]
    datasets = ["proverqa", "justlogic", "folio"]

    # Plot comparison grid (Figure 2)
    if not results_df.empty:
        plot_comparison_grid(results_df, file_ids=file_ids, datasets=datasets, y_min=0, y_max=100)

    # Inference frequency plots (CoT-only patterns) (Figure 4)
    cot_only_patterns = {
        "proverqa": os.path.join(BASE_DIR, "proverqa_*_output_sample_32_orm_cot-gpt4o-sample8.jsonl"),
        "folio": os.path.join(BASE_DIR, "folio_*_output_sample_32_orm_cot-gpt4o-sample10.jsonl"),
        # "justlogic": os.path.join(BASE_DIR, "justlogic_*_output_sample_32_orm_cot-gpt4o-sample8.jsonl"),
    }
    MODELS_DICT = {
        "qwen25": "Qwen2.5 7B",
        "qwen3": "Qwen3 8B",
        "gpt4o": "GPT-4o",
        "llama31": "LLaMA v3.1 8B",
        "gemma1": "Gemma3 1B",
        "gemma4": "Gemma3 4B",
        "gemma12": "Gemma3 12B",
    }
    allowed_ids = {"qwen25", "gpt4o", "llama31", "qwen3"}
    plot_inference_frequency(
        folder_path=BASE_DIR,
        batch_sizes=BATCH_SIZES,
        patterns=cot_only_patterns,
        models_dict=MODELS_DICT,
        allowed_file_ids=allowed_ids,
    )
