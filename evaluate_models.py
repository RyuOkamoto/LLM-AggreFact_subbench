import csv
from os import getenv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import nltk
import numpy as np
from datasets import load_from_disk
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm

from scorer import (
    FactCGDeBERTa,
    MiniCheckDeBERTa,
    MiniCheckFlanT5,
    MiniCheckRoBERTa,
    # TRUE_T5,
    PreTrainedRoBERTa,
    Scorer,
)

load_dotenv()
HF_TOKEN = getenv("HF_TOKEN")
login(token=HF_TOKEN, add_to_git_credential=True)

nltk.download("punkt")

SUBSET_PATH = "LLM-AggreFact_subset"
POSITIVE_LABEL = 1

OUTPUT_DIR = Path("./results") / "with_sliding_stride=3"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    llm_aggrefact_subset = load_from_disk(SUBSET_PATH).shuffle(seed=42)

    scoreres: Scorer = [
        FactCGDeBERTa(),
        MiniCheckDeBERTa(),
        MiniCheckFlanT5(),
        MiniCheckRoBERTa(),
        # TRUE_T5(),  very huge!!
        PreTrainedRoBERTa(),
    ]

    for scorer in scoreres:
        ys_true = []
        ys_score = []
        for example in tqdm(llm_aggrefact_subset, desc=scorer.model_name):
            doc = example["doc"]
            claim = example["claim"]
            y_true = example["label"]
            y_score = scorer.score(doc, claim)
            ys_true.append(y_true)
            ys_score.append(y_score)
        save_results(scorer.model_name, ys_true, ys_score)


def save_results(model_name: str, ys_true: List[int], ys_score: List[float]):
    thresholds = np.linspace(0, 1, 1000)
    tps, fns, fps, tns = calculate_tps_fns_fps_tns(ys_true, ys_score, thresholds)
    baccs, tprs, fprs, precisions = calculate_all_metrics(tps, fns, fps, tns)

    best_idx = np.argmax(baccs)
    best_bacc = baccs[best_idx]
    best_threshold = thresholds[best_idx]
    best_tpr = tprs[best_idx]
    best_fpr = fprs[best_idx]
    best_precision = precisions[best_idx]

    visualize_curve(
        save_dir=OUTPUT_DIR / "BAcc",
        save_name=model_name,
        xs=thresholds,
        ys=baccs,
        title="Balanced Accuracy Curve",
        xlabel="Threshold",
        ylabel="Balanced Accuracy",
        annotations=[(best_threshold, best_bacc)],
    )

    visualize_curve(
        save_dir=OUTPUT_DIR / "ROC",
        save_name=model_name,
        xs=fprs,
        ys=tprs,
        title="ROC Curve",
        xlabel="FPR",
        ylabel="TPR",
        annotations=[(best_fpr, best_tpr)],
    )

    visualize_curve(
        save_dir=OUTPUT_DIR / "PR",
        save_name=model_name,
        xs=tprs,
        ys=precisions,
        title="Precision-Recall Curve",
        xlabel="Recall",
        ylabel="Precision",
        annotations=[(best_tpr, best_precision)],
    )

    csv_path = OUTPUT_DIR / f"{model_name}.csv"
    with csv_path.open("w") as f:
        csv.writer(f).writerow(["y_true", "y_score"])
        for y_true, y_score in zip(ys_true, ys_score):
            csv.writer(f).writerow([y_true, y_score])


def calculate_tps_fns_fps_tns(
    ys_true: List[int], ys_score: List[float], thresholds: List[float]
):
    tps, fns, fps, tns = [], [], [], []
    for t in thresholds:
        ys_pred = (np.array(ys_score) >= t).astype(int)
        tp = sum((np.array(ys_true) == POSITIVE_LABEL) & (ys_pred == POSITIVE_LABEL))
        fn = sum((np.array(ys_true) == POSITIVE_LABEL) & (ys_pred != POSITIVE_LABEL))
        fp = sum((np.array(ys_true) != POSITIVE_LABEL) & (ys_pred == POSITIVE_LABEL))
        tn = sum((np.array(ys_true) != POSITIVE_LABEL) & (ys_pred != POSITIVE_LABEL))
        tps.append(tp)
        fns.append(fn)
        fps.append(fp)
        tns.append(tn)
    return np.array(tps), np.array(fns), np.array(fps), np.array(tns)


def calculate_all_metrics(tps, fns, fps, tns):
    actual_positives = tps + fns
    actual_negatives = tns + fps
    predicted_positives = tps + fps

    with np.errstate(divide="ignore", invalid="ignore"):
        tpr = np.where(actual_positives != 0, tps / actual_positives, 0.0)
        tnr = np.where(actual_negatives != 0, tns / actual_negatives, 0.0)
        fpr = np.where(actual_negatives != 0, fps / actual_negatives, 0.0)
        precisions = np.where(predicted_positives != 0, tps / predicted_positives, 0.0)

    baccs = (tpr + tnr) / 2
    return baccs, tpr, fpr, precisions


def visualize_curve(save_dir: Path, save_name: str, xs, ys, **kwgs):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x=xs, y=ys, s=20, c="steelblue", alpha=0.8, zorder=1)
    ax.set_title(kwgs.get("title", ""))
    ax.set_xlabel(kwgs.get("xlabel", ""))
    ax.set_ylabel(kwgs.get("ylabel", ""))
    for x, y in kwgs.get("annotations", []):
        ax.axvline(x=x, color="grey", linestyle="--", zorder=2)
        ax.axhline(y=y, color="grey", linestyle="--", zorder=2)
        ax.scatter(x=[x], y=[y], s=20, c="firebrick", alpha=0.8, zorder=3)
        ax.text(x + 0.01, 0.01, f"{x:.4f}", ha="left", va="bottom", zorder=3)
        ax.text(x + 0.01, y + 0.01, f"{y:.4f}", ha="left", va="bottom", zorder=3)
    ax.grid()
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()
    save_dir.mkdir(exist_ok=True)
    png_path = save_dir / f"{save_name}.png"
    pdf_path = save_dir / f"{save_name}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)


if __name__ == "__main__":
    main()
