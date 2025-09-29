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
    def __init__(
        self, model_path: str, max_model_len: int, device: str, batch_size: int
    ):
        self.model_path = model_path
        self.model_name = self.model_path.split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, revision="main"
        )
        self.max_model_len = max_model_len

        # reserve some space (hard coded, 200) for the claim to be checked
        self.default_chunk_size = self.max_model_len - 200
        assert self.default_chunk_size > 0

        self.device = device
        self.batch_size = batch_size

    def score(self, doc: str, claim: str) -> float:
        doc_chunk_gen = generate_chunks(doc, self.default_chunk_size)
        max_entailment_prob = 0.0
        for doc_batch in batched(doc_chunk_gen, self.batch_size):
            doc_batch = list(doc_batch)
            batch_input = self._batch_tokenize(doc_batch, claim)
            batch_input = {k: v.to(self.device) for k, v in batch_input.items()}
            with torch.no_grad():
                probs = self._infer_batch(batch_input)
                current_max_prob = probs.max().item()
            if current_max_prob > max_entailment_prob:
                max_entailment_prob = current_max_prob
        return max_entailment_prob

    @abstractmethod
    def _infer_batch(self, batch_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    @abstractmethod
    def _concat_doc_and_claim(self, doc, claim):
        pass

    def _batch_tokenize(self, doc_batch, claim):
        return self.tokenizer(
            [self._concat_doc_and_claim(doc, claim) for doc in doc_batch],
            max_length=self.max_model_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )


class XBERTForNLI(Scorer):
    def __init__(
        self,
        model_path: str,
        max_model_len: int,
        label2id: Dict,
        device: str,
        batch_size: int,
    ):
        super().__init__(model_path, max_model_len, device, batch_size)
        config = AutoConfig.from_pretrained(
            model_path,
            label2id=label2id,
            finetuning_task="text-classification",
            revision="main",
        )
        config.num_labels = len(set(label2id.values()))
        config.problem_type = "single_label_classification"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=config,
            revision="main",
            ignore_mismatched_sizes=False,
        )
        self.entailment_label = label2id.get("entailment", 0)
        self.model.to(device)
        self.model.eval()

    @override
    def _infer_batch(self, batch_input):
        logits = self.model(**batch_input).logits.cpu()
        label_probs = torch.nn.functional.softmax(logits, dim=1)
        return label_probs[:, self.entailment_label]

    @override
    def _concat_doc_and_claim(self, doc, claim):
        eos_token = self.tokenizer.eos_token
        return eos_token.join([doc, claim])


class MiniCheckDeBERTa(XBERTForNLI):
    def __init__(
        self,
        model_path: str = "lytang/MiniCheck-DeBERTa-v3-Large",
        max_model_len: int = 2048,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE // 4,
    ):
        label2id = {"not_entailment": 0, "entailment": 1}
        super().__init__(model_path, max_model_len, label2id, device, batch_size)


# ref: https://github.com/derenlei/FactCG/blob/main/factcg/inference.py
class FactCGDeBERTa(XBERTForNLI):
    def __init__(
        self,
        model_path: str = "yaxili96/FactCG-DeBERTa-v3-Large",
        max_model_len: int = 2048,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE // 4,
    ):
        label2id = {"not_entailment": 0, "entailment": 1}
        super().__init__(model_path, max_model_len, label2id, device, batch_size)

    @override
    def _concat_doc_and_claim(self, doc, claim):
        return (
            "{text_a}\n"
            "\n"
            'Choose your answer: based on the paragraph above can we conclude that "{text_b}"?\n'
            "\n"
            "OPTIONS:\n"
            "- Yes\n"
            "- No\n"
            "I think the answer is ".format(text_a=doc, text_b=claim)
        )


class PreTrainedRoBERTa(XBERTForNLI):
    def __init__(
        self,
        model_path: str = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
        max_model_len: int = 512,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
        super().__init__(model_path, max_model_len, label2id, device, batch_size)


class MiniCheckRoBERTa(XBERTForNLI):
    def __init__(
        self,
        model_path: str = "lytang/MiniCheck-RoBERTa-Large",
        max_model_len: int = 512,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        label2id = {"not_entailment": 0, "entailment": 1}
        super().__init__(model_path, max_model_len, label2id, device, batch_size)


# ref: https://github.com//Liyan06/MiniCheck/blob/main/minicheck/inference.py
class MiniCheckFlanT5(Scorer):
    def __init__(
        self,
        model_path: str = "lytang/MiniCheck-Flan-T5-Large",
        max_model_len: int = 2048,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE // 4,
    ):
        super().__init__(model_path, max_model_len, device, batch_size)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        self.model.to(device)
        self.model.eval()

    @override
    def _infer_batch(self, batch_input):
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
        return label_probs[:, 1]

    @override
    def _concat_doc_and_claim(self, doc, claim):
        return "predict: " + self.tokenizer.eos_token.join([doc, claim])
