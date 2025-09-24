from os import getenv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import nltk
import numpy as np
import seaborn as sns
from datasets import load_from_disk
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm

from scorer import (
    MiniCheckDeBERTa,
    MiniCheckFlanT5,
    MiniCheckRoBERTa,
    OurTrainedRoBERTa,
    PreTrainedRoBERTa,
    Scorer,
)

load_dotenv()
HF_TOKEN = getenv("HF_TOKEN")
login(token=HF_TOKEN, add_to_git_credential=True)

nltk.download("punkt")

SUBSET_PATH = "LLM-AggreFact_subset"
POSITIVE_LABEL = 1

OUTPUT_DIR = Path("./results")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    llm_aggrefact_subset = load_from_disk(SUBSET_PATH)

    scoreres: Scorer = [
        MiniCheckDeBERTa(),
        OurTrainedRoBERTa(),
        PreTrainedRoBERTa(),
        MiniCheckRoBERTa(),
        MiniCheckFlanT5(),
    ]

    for scorer in scoreres:
        ys_true = []
        ys_score = []

        for example in tqdm(llm_aggrefact_subset):
            doc = example["doc"]
            claim = example["claim"]
            y_true = example["label"]
            y_score = scorer.score(doc, claim)
            ys_true.append(y_true)
            ys_score.append(y_score)

        csv = "y_true,y_score\n"
        for y_true, y_score in zip(ys_true, ys_score):
            csv += f"{y_true},{y_score}\n"
        csv_path = OUTPUT_DIR / f"{scorer.model_name}.csv"
        csv_path.write_text(csv)

        thresholds = np.linspace(0, 1, 1000)
        tps, fns, fps, tns = calculate_tps_fns_fps_tns(ys_true, ys_score, thresholds)
        baccs, tprs, fprs, precisions = calculate_all_metrics(tps, fns, fps, tns)

        best_idx = np.argmax(baccs)
        best_bacc = baccs[best_idx]
        best_threshold = thresholds[best_idx]
        best_tpr = tprs[best_idx]
        best_fpr = fprs[best_idx]
        best_precision = precisions[best_idx]

        save_curve(
            save_dir=OUTPUT_DIR / "BAcc",
            save_name=scorer.model_name,
            xs=thresholds,
            ys=baccs,
            title="Balanced Accuracy Curve",
            xlabel="Threshold",
            ylabel="Balanced Accuracy",
            annotations=[(best_threshold, best_bacc)],
        )

        save_curve(
            save_dir=OUTPUT_DIR / "ROC",
            save_name=scorer.model_name,
            xs=fprs,
            ys=tprs,
            title="ROC Curve",
            xlabel="FPR",
            ylabel="TPR",
            annotations=[(best_fpr, best_tpr)],
        )

        save_curve(
            save_dir=OUTPUT_DIR / "PR",
            save_name=scorer.model_name,
            xs=tprs,
            ys=precisions,
            title="Precision-Recall Curve",
            xlabel="Recall",
            ylabel="Precision",
            annotations=[(best_tpr, best_precision)],
        )

        best_tp = tps[best_idx]
        best_fn = fns[best_idx]
        best_fp = fps[best_idx]
        best_tn = tns[best_idx]
        best_fnr = best_fn / (best_fn + best_tp) if (best_fn + best_tp) > 0 else 0.0
        best_tnr = best_tn / (best_tn + best_fp) if (best_tn + best_fp) > 0 else 0.0
        save_path = OUTPUT_DIR / f"{scorer.model_name}_summary.csv"
        save_path.write_text(
            "TP,FN,FP,TN\n"
            f"{best_tp},{best_fn},{best_fp},{best_tn}\n"
            f"{best_tpr:.4f},{best_fnr:.4f},{best_fpr:.4f},{best_tnr:.4f}\n"
        )


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


def save_curve(save_dir: Path, save_name: str, xs, ys, **kwgs):
    fig, ax = plt.subplots(figsize=(5, 5))
    color = kwgs.get("color", "blue")
    sns.lineplot(x=xs, y=ys, color=color)
    ax.set_title(kwgs.get("title", ""))
    ax.set_xlabel(kwgs.get("xlabel", ""))
    ax.set_ylabel(kwgs.get("ylabel", ""))
    for x, y in kwgs.get("annotations", []):
        ax.axvline(x=x, color="grey", linestyle="--")
        ax.axhline(y=y, color="grey", linestyle="--")
        ax.text(x + 0.01, 0.01, f"{x:.4f}", ha="left", va="bottom")
        ax.text(x + 0.01, y + 0.01, f"{y:.4f}", ha="left", va="bottom")
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
