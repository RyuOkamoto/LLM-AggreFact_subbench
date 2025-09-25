from abc import ABC, abstractmethod
from itertools import batched
from typing import Dict, Generator, List, override

import torch
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_BATCH_SIZE = 4


def generate_chunks(text: str, chunk_size: int) -> Generator[str, None, None]:
    sentences = sent_tokenize_with_newlines(text) or [""]
    current_chunks = []
    current_word_count = 0
    for sentence in sentences:
        sentence_word_count = len(word_tokenize(sentence))
        if current_word_count + sentence_word_count > chunk_size:
            chunk = " ".join(current_chunks).replace(" \n ", "\n").strip()
            if chunk:
                yield chunk
            current_chunks = [sentence]
            current_word_count = sentence_word_count
        else:
            current_chunks.append(sentence)
            current_word_count += sentence_word_count
    if current_chunks:
        chunk = " ".join(current_chunks).replace(" \n ", "\n").strip()
        if chunk:
            yield chunk


def sent_tokenize_with_newlines(text: str) -> List[str]:
    blocks = [sent_tokenize(block) for block in text.split("\n")]
    sentences = []
    for block in blocks:
        sentences.extend(block)
        sentences.append("\n")
    return sentences[:-1]


# wrapper class for model to be evaluated
class Scorer(ABC):
    @property
    @abstractmethod
    def model_path(self) -> str:
        pass

    @property
    def model_name(self) -> str:
        return self.model_path.split("/")[-1]

    @abstractmethod
    def score(self, doc: str, claim: str) -> float:
        pass


class XBERTForNLI(Scorer):
    def __init__(
        self,
        max_model_len: int,
        label2id: Dict,
        device: str,
        batch_size: int,
    ):
        self.max_model_len = max_model_len
        config = AutoConfig.from_pretrained(
            self.model_path,
            label2id=label2id,
            finetuning_task="text-classification",
            revision="main",
        )
        config.num_labels = len(set(label2id.values()))
        config.problem_type = "single_label_classification"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=True, revision="main"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            config=config,
            revision="main",
            ignore_mismatched_sizes=False,
        )
        self.entailment_label = label2id.get("entailment", 1)
        self.device = device
        self.batch_size = batch_size
        self.default_chunk_size = (
            self.max_model_len - 200
        )  # reserve some space (hard coded) for the claim to be checked
        assert self.default_chunk_size > 0
        self.model.to(device)
        self.model.eval()

    @override
    def score(self, doc: str, claim: str) -> float:
        doc_chunk_gen = generate_chunks(doc, self.default_chunk_size)
        max_entailment_prob = 0.0
        for doc_batch in batched(doc_chunk_gen, self.batch_size):
            doc_batch = list(doc_batch)
            batch_input = self._batch_tokenize(doc_batch, claim)
            batch_input = {k: v.to(self.device) for k, v in batch_input.items()}
            with torch.no_grad():
                logits = self.model(**batch_input).logits.cpu()
                label_probs = torch.nn.functional.softmax(logits, dim=1)
                current_max_prob = label_probs[:, self.entailment_label].max().item()
            if current_max_prob > max_entailment_prob:
                max_entailment_prob = current_max_prob
        return max_entailment_prob

    def _batch_tokenize(self, doc_batch, claim):
        eos_token = self.tokenizer.eos_token
        original_text = [eos_token.join([doc, claim]) for doc in doc_batch]
        return self.tokenizer(
            original_text,
            max_length=self.max_model_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )


class DeBERTaForNLI(XBERTForNLI):
    def __init__(
        self,
        label2id: Dict,
        device: str,
        batch_size: int,
    ):
        max_model_len = 2048
        super().__init__(
            max_model_len, label2id=label2id, device=device, batch_size=batch_size
        )


class RoBERTaForNLI(XBERTForNLI):
    def __init__(
        self,
        label2id: Dict,
        device: str,
        batch_size: int,
    ):
        max_model_len = 512
        super().__init__(
            max_model_len, label2id=label2id, device=device, batch_size=batch_size
        )


class MiniCheckDeBERTa(DeBERTaForNLI):
    model_path = "lytang/MiniCheck-DeBERTa-v3-Large"

    def __init__(
        self,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE // 4,
    ):
        label2id = {"not_entailment": 0, "entailment": 1}
        super().__init__(label2id, device, batch_size)


class OurTrainedRoBERTa(RoBERTaForNLI):
    model_path = "./our_trained_roberta_large"

    def __init__(
        self,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
        super().__init__(label2id, device, batch_size)


class PreTrainedRoBERTa(RoBERTaForNLI):
    model_path = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"

    def __init__(
        self,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
        super().__init__(label2id, device, batch_size)


class MiniCheckRoBERTa(RoBERTaForNLI):
    model_path = "lytang/MiniCheck-RoBERTa-Large"

    def __init__(
        self,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        label2id = {"not_entailment": 0, "entailment": 1}
        super().__init__(label2id, device, batch_size)


class MiniCheckFlanT5(Scorer):
    model_path = "lytang/MiniCheck-Flan-T5-Large"

    def __init__(
        self, device: str = DEFAULT_DEVICE, batch_size: int = DEFAULT_BATCH_SIZE // 4
    ):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.max_model_len = 2048
        self.max_output_len = 256
        self.device = device
        self.batch_size = batch_size
        self.default_chunk_size = (
            self.max_model_len - 200
        )  # reserve some space (hard coded) for the claim to be checked
        self.model.to(device)
        self.model.eval()

    @override
    def score(self, doc: str, claim: str) -> float:
        doc_chunk_gen = generate_chunks(doc, self.default_chunk_size)
        max_support_prob = 0.0
        for doc_batch in batched(doc_chunk_gen, self.batch_size):
            doc_batch = list(doc_batch)
            batch_input = self._batch_tokenize(doc_batch, claim)
            batch_input = {k: v.to(self.device) for k, v in batch_input.items()}
            with torch.no_grad():
                decoder_input_ids = torch.zeros(
                    (batch_input["input_ids"].size(0), 1), dtype=torch.long
                ).to(self.device)
                outputs = self.model(
                    input_ids=batch_input["input_ids"],
                    attention_mask=batch_input["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                )
                logits = outputs.logits.squeeze(1)

                # 3 for no support and 209 for support
                label_logits = logits[:, torch.tensor([3, 209])].cpu()
                label_probs = torch.nn.functional.softmax(label_logits, dim=-1)
                current_max_prob = label_probs[:, 1].max().item()

            if current_max_prob > max_support_prob:
                max_support_prob = current_max_prob
        return max_support_prob

    def _batch_tokenize(self, doc_batch, claim):
        eos_token = self.tokenizer.eos_token
        original_text = [eos_token.join([doc, claim]) for doc in doc_batch]
        return self.tokenizer(
            ["predict: " + text for text in original_text],
            max_length=self.max_model_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
