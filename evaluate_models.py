from os import getenv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import nltk
import numpy as np
import seaborn as sns
import torch
from datasets import load_from_disk
from dotenv import load_dotenv
from huggingface_hub import login
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from scorer import MiniCheck, RoBERTaForNLI

load_dotenv()
HF_TOKEN = getenv("HF_TOKEN")
login(token=HF_TOKEN, add_to_git_credential=True)

nltk.download("punkt")

SUBSET_SAVE_PATH = "LLM-AggreFact_subset"
DATASET_POSITIVE_LABEL = 1

MAX_SENTENCES = 5  # L in the formular for RQ1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_BATCH_SIZE = 16

OUTPUT_DIR = Path("./results")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    llm_aggrefact_subset = load_from_disk(
        SUBSET_SAVE_PATH
    )  # .shuffle(seed=42).select(range(100))

    # our fine-tunined NLI model
    our_trained_roberta = RoBERTaForNLI(
        model_path_or_name="./3nli_trained_roberta_large",
        max_sentences=MAX_SENTENCES,
        device=DEVICE,
        batch_size=EVAL_BATCH_SIZE,
    )

    # pretrained NLI model
    # see: https://huggingface.co/ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli
    pretrained_roberta = RoBERTaForNLI(
        model_path_or_name="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
        max_sentences=MAX_SENTENCES,
        device=DEVICE,
        batch_size=EVAL_BATCH_SIZE,
    )

    # flan-T5 MiniCheck
    minicheck_flan_T5 = MiniCheck(
        model_path_or_name="lytang/MiniCheck-Flan-T5-Large",
        device=DEVICE,
        batch_size=EVAL_BATCH_SIZE,
    )

    # roberta-large MiniCheck
    minicheck_roberta = MiniCheck(
        model_path_or_name="lytang/MiniCheck-RoBERTa-Large",
        device=DEVICE,
        batch_size=EVAL_BATCH_SIZE,
    )

    # # deberta-large MiniCheck
    # minicheck_deberta = MiniCheck(
    #     model_path_or_name="lytang/MiniCheck-DeBERTa-v3-Large",
    #     device=DEVICE,
    #     batch_size=1,  # large input size
    # )

    models = [
        ("our_trained_roberta", our_trained_roberta),
        ("pretrained_roberta", pretrained_roberta),
        ("minicheck_flan_T5", minicheck_flan_T5),
        ("minicheck_roberta", minicheck_roberta),
        # ("minicheck_deberta", minicheck_deberta),
    ]

    for model_name, model in models:
        ys_true = []
        ys_score = []
        for example in tqdm(llm_aggrefact_subset):
            doc = example["doc"]
            claim = example["claim"]
            y_true = example["label"]
            y_score = model.score(doc, claim)
            ys_true.append(y_true)
            ys_score.append(y_score)
        if isinstance(model, RoBERTaForNLI):
            model_name += f"(L={model.max_sentences})"
        visualize_roc_pr_curves(model_name, ys_true, ys_score)
        visualize_bacc_curve(model_name, ys_true, ys_score)


def visualize_roc_pr_curves(model_name: str, ys_true: List[int], ys_score: List[float]):
    fpr, tpr, _ = roc_curve(ys_true, ys_score, pos_label=DATASET_POSITIVE_LABEL)
    ps, rs, _ = precision_recall_curve(
        ys_true, ys_score, pos_label=DATASET_POSITIVE_LABEL
    )
    auc = roc_auc_score(ys_true, ys_score)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    sns.lineplot(x=fpr, y=tpr, color="blue", ax=axes[0])
    axes[0].set_title(f"ROC Curve (AUC = {auc:.2f})")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].grid()
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    sns.lineplot(x=rs, y=ps, color="orange", ax=axes[1])
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].grid()
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    png_path = OUTPUT_DIR / f"roc_pr_curves_{model_name}.png"
    pdf_path = OUTPUT_DIR / f"roc_pr_curves_{model_name}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)


def visualize_bacc_curve(model_name: str, ys_true: List[int], ys_score: List[float]):
    thresholds = np.linspace(0, 1, 1000)
    baccs = []
    for t in thresholds:
        ys_pred = (np.array(ys_score) >= t).astype(int)
        bacc = balanced_accuracy_score(ys_true, ys_pred)
        baccs.append(bacc)

    best_bacc = max(baccs)
    best_threshold = thresholds[np.argmax(baccs)]

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.lineplot(x=thresholds, y=baccs, color="blue")
    ax.set_title("Balanced Accuracy Curve")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Balanced Accuracy")
    ax.grid()
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.axvline(x=best_threshold, color="grey", linestyle="--")
    ax.axhline(y=best_bacc, color="grey", linestyle="--")
    ax.text(
        best_threshold + 0.01, 0.01, f"{best_threshold:.3f}", ha="left", va="bottom"
    )
    ax.text(
        best_threshold + 0.01,
        best_bacc + 0.01,
        f"{best_bacc:.3f}",
        ha="left",
        va="bottom",
    )
    png_path = OUTPUT_DIR / f"bacc_curves_{model_name}.png"
    pdf_path = OUTPUT_DIR / f"bacc_curves_{model_name}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)


if __name__ == "__main__":
    main()
