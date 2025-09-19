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

OUTPUT_DIR = Path("./results")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    llm_aggrefact_subset = concatenate_datasets(
        [
            load_dataset(HF_DATASET_PATH, split="dev"),
            load_dataset(HF_DATASET_PATH, split="test"),
        ]
    ).filter(
        is_claim_composed_of_one_sentence, batched=True, batch_size=512, num_proc=4
    )
    llm_aggrefact_subset.save_to_disk(SUBSET_SAVE_PATH, num_proc=4)
    visualize_distribution(llm_aggrefact_subset)


def is_claim_composed_of_one_sentence(batch: Dict[str, List]) -> List[bool]:
    claims = batch["claim"]
    return [len(sent_tokenize(claim)) == 1 for claim in claims]


def visualize_distribution(dataset: Dataset):
    assert "doc" in dataset.column_names and "claim" in dataset.column_names
    result = dataset.map(
        count_words_and_sentences, batched=True, batch_size=512, num_proc=4
    )
    doc_sents_nums = result["doc_sents_nums"]
    doc_words_nums = result["doc_words_nums"]
    claim_words_nums = result["claim_words_nums"]

    def drop_top_percent(data, percent=1):
        cutoff = np.percentile(data, 100 - percent)
        return [x for x in data if x <= cutoff]

    doc_sents_nums = drop_top_percent(doc_sents_nums, percent=1)
    doc_words_nums = drop_top_percent(doc_words_nums, percent=1)
    claim_words_nums = drop_top_percent(claim_words_nums, percent=1)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    sns.histplot(doc_sents_nums, ax=axes[0], bins=50, kde=True)
    sns.histplot(doc_words_nums, ax=axes[1], bins=50, color="orange", kde=True)
    sns.histplot(claim_words_nums, ax=axes[2], bins=50, color="orange", kde=True)
    axes[0].set_title("Document Sentence Count Distribution ($\\leq$ 99%)")
    axes[1].set_title("Document Word Count Distribution ($\\leq$ 99%)")
    axes[2].set_title("Claim Word Count Distribution ($\\leq$ 99%)")
    png_path = OUTPUT_DIR / "data_distribution.png"
    pdf_path = OUTPUT_DIR / "data_distribution.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)


def count_words_and_sentences(batch: Dict[str, List]) -> Dict[str, List[int]]:
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
