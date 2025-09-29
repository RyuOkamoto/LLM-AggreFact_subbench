from os import getenv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import nltk
import numpy as np
import seaborn as sns
from datasets import Dataset, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from nltk.tokenize import sent_tokenize, word_tokenize

load_dotenv()
HF_TOKEN = getenv("HF_TOKEN")
login(token=HF_TOKEN, add_to_git_credential=True)

nltk.download("punkt")

HF_DATASET_PATH = "lytang/LLM-AggreFact"
SUBSET_SAVE_PATH = "LLM-AggreFact_subset"

OUTPUT_DIR = Path("./results") / "statistics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    llm_aggrefact_subset = (
        concatenate_datasets(
            [
                load_dataset(HF_DATASET_PATH, split="dev"),
                load_dataset(HF_DATASET_PATH, split="test"),
            ]
        )
        .filter(
            is_claim_composed_of_one_sentence, batched=True, batch_size=512, num_proc=4
        )
        .filter(
            is_word_count_between_1k_and_10k, batched=True, batch_size=512, num_proc=4
        )
    )
    llm_aggrefact_subset.save_to_disk(SUBSET_SAVE_PATH, num_proc=4)
    save_statistices(llm_aggrefact_subset)


def is_claim_composed_of_one_sentence(batch: Dict[str, List]) -> List[bool]:
    claims = batch["claim"]
    return [len(sent_tokenize(claim)) == 1 for claim in claims]


def is_word_count_between_1k_and_10k(batch: Dict[str, List]) -> List[bool]:
    docs = batch["doc"]
    return [
        len(word_tokenize(doc)) >= 1_000 and len(word_tokenize(doc)) < 10_000
        for doc in docs
    ]


def save_statistices(dataset: Dataset):
    visualize_from_dataset_pie(dataset)
    visualize_word_count_distribution(dataset)
    visualize_sent_count_distribution(dataset)


def visualize_from_dataset_pie(dataset: Dataset):
    from_count = {}
    for from_dataset in dataset["dataset"]:
        from_count[from_dataset] = from_count.get(from_dataset, 0) + 1

    def my_format(pct):
        total = sum(from_count.values())
        val = int(round(pct * total / 100.0))
        return "{v:,}".format(v=val)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = sns.color_palette("pastel")
    _, texts, autotexts = ax.pie(
        from_count.values(),
        labels=from_count.keys(),
        explode=[0.025] * len(from_count),
        colors=colors,
        autopct=my_format,
        wedgeprops=dict(width=0.75),
    )
    for text, autotext in zip(texts, autotexts):
        if "AggreFact-XSum" in text.get_text():
            x, y = text.get_position()
            text.set_position((x, y + 0.05))
            autotext.set_y(y + 0.025)
        if "AggreFact-CNN" in text.get_text():
            x, y = text.get_position()
            text.set_position((x, y - 0.05))
            autotext.set_y(y - 0.025)
    ax.set_title("From Dataset Pie")
    png_path = OUTPUT_DIR / "from_dataset_pie.png"
    pdf_path = OUTPUT_DIR / "from_dataset_pie.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)


def visualize_word_count_distribution(dataset: Dataset):
    dist = dataset.map(count_words_and_sents, batched=True, batch_size=512, num_proc=4)
    doc_dist = dist["doc_words_nums"]
    claim_dist = dist["claim_words_nums"]
    doc_mean = np.mean(doc_dist)
    claim_mean = np.mean(claim_dist)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    sns.histplot(doc_dist, ax=axes[0], bins=50, color="steelblue", kde=True)
    axes[0].axvline(x=doc_mean, color="#36648B", linestyle="--")
    axes[0].text(
        doc_mean + 100,
        axes[0].get_ylim()[1] - 100,
        f"mean={int(doc_mean):,}",
        ha="left",
        va="top",
        zorder=3,
    )
    sns.histplot(claim_dist, ax=axes[1], bins=50, color="orange", kde=True)
    axes[1].axvline(x=claim_mean, color="#CC8400", linestyle="--")
    axes[1].text(
        claim_mean + 3,
        axes[1].get_ylim()[1] - 40,
        f"mean={int(claim_mean):,}",
        ha="left",
        va="top",
        zorder=3,
    )
    axes[0].set_title("Doc # of Words Distribution")
    axes[1].set_title("Claim # of Words Distribution")
    png_path = OUTPUT_DIR / "word_count_distribution.png"
    pdf_path = OUTPUT_DIR / "word_count_distribution.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 5))
    sns.boxplot(data=doc_dist, ax=axes[0], color="steelblue")
    sns.boxplot(data=claim_dist, ax=axes[1], color="orange")
    axes[0].set_title("Doc # of Words Distribution")
    axes[1].set_title("Claim # of Words Distribution")
    png_path = OUTPUT_DIR / "word_count_boxplot.png"
    pdf_path = OUTPUT_DIR / "word_count_boxplot.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)


def visualize_sent_count_distribution(dataset: Dataset):
    dist = dataset.map(count_words_and_sents, batched=True, batch_size=512, num_proc=4)
    doc_dist = dist["doc_sents_nums"]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(doc_dist, ax=ax, bins=50, color="steelblue", kde=True)
    ax.set_title("Doc # of Sentences Distribution")
    png_path = OUTPUT_DIR / "sent_count_distribution.png"
    pdf_path = OUTPUT_DIR / "sent_count_distribution.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)


def count_words_and_sents(batch: Dict[str, List]) -> Dict[str, List[int]]:
    docs = batch["doc"]
    claims = batch["claim"]
    doc_sents_nums = [len(sent_tokenize(doc)) for doc in docs]
    doc_words_nums = [len(word_tokenize(doc)) for doc in docs]
    claim_words_nums = [len(word_tokenize(claim)) for claim in claims]
    return {
        "doc_sents_nums": doc_sents_nums,
        "doc_words_nums": doc_words_nums,
        "claim_words_nums": claim_words_nums,
    }


if __name__ == "__main__":
    main()
