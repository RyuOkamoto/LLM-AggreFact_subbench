from abc import ABC, abstractmethod
from typing import Dict, override

import torch
from nltk.tokenize import sent_tokenize
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_BATCH_SIZE = 16


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


def sent_tokenize_with_newlines(text):
    blocks = text.split("\n")
    tokenized_blocks = [sent_tokenize(block) for block in blocks]
    tokenized_text = []
    for block in tokenized_blocks:
        tokenized_text.extend(block)
        tokenized_text.append("\n")
    return tokenized_text[:-1]


class XBERTForNLI(Scorer):
    def __init__(
        self,
        max_model_len: int,
        label2id: Dict,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.max_model_len = max_model_len
        config = AutoConfig.from_pretrained(
            self.model_path,
            label2id=label2id,
            finetuning_task="text-classification",
            revision="main",
            token=None,
        )
        config.problem_type = "single_label_classification"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=True, revision="main", token=None
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            config=config,
            revision="main",
            token=None,
            ignore_mismatched_sizes=False,
        )
        self.pos_label = label2id.get("entailment", 1)
        self.device = device
        self.batch_size = batch_size
        self.default_chunk_size = (
            self.max_model_len - 300
        )  # reserve some space (hard coded) for the claim to be checked
        assert self.default_chunk_size > 0
        self.model.to(device)
        self.model.eval()

    @override
    def score(self, doc: str, claim: str) -> float:
        def chunks(lst, n):
            current_chunk = []
            current_token_count = 0
            for sentence in lst:
                sentence_word_count = len(
                    self.tokenizer(
                        sentence,
                        padding=False,
                        add_special_tokens=False,
                        max_length=self.max_model_len,
                        truncation=True,
                    )["input_ids"]
                )
                if current_token_count + sentence_word_count > n:
                    yield " ".join(current_chunk)
                    current_chunk = [sentence]
                    current_token_count = sentence_word_count
                else:
                    current_chunk.append(sentence)
                    current_token_count += sentence_word_count
            if current_chunk:
                yield " ".join(current_chunk)

        doc_sents = sent_tokenize_with_newlines(doc)
        doc_sents = doc_sents or [""]
        doc_chunks = [
            chunk.replace(" \n ", "\n").strip()
            for chunk in chunks(doc_sents, self.default_chunk_size)
        ]
        doc_chunks = [chunk for chunk in doc_chunks if chunk != ""]
        claim_repeat = [claim] * len(doc_chunks)
        output = self._inference(doc_chunks, claim_repeat)
        return output

    def _inference(self, doc, claim):
        if isinstance(doc, str) and isinstance(claim, str):
            doc = [doc]
            claim = [claim]

        batch_input, _ = self._batch_tokenize(doc, claim)
        label_probs_list = []
        for mini_batch_input in batch_input:
            mini_batch_input = {
                k: v.to(self.device) for k, v in mini_batch_input.items()
            }
            with torch.no_grad():
                outputs = self.model(**mini_batch_input)
                logits = outputs.logits
                label_probs = torch.nn.functional.softmax(logits, dim=1)
                label_probs_list.append(label_probs)
        label_probs = torch.cat(label_probs_list)
        max_support_prob = label_probs[:, self.pos_label].max().item()
        return max_support_prob

    def _batch_tokenize(self, doc, claim):
        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        original_text = [
            self.tokenizer.eos_token.join([one_doc, one_claim])
            for one_doc, one_claim in zip(doc, claim)
        ]
        batch_input = []
        batch_concat_text = []
        for mini_batch in chunks(original_text, self.batch_size):
            model_inputs = self.tokenizer(
                [text for text in mini_batch],
                max_length=self.max_model_len,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            batch_input.append(model_inputs)
            batch_concat_text.append(mini_batch)
        return batch_input, batch_concat_text


class DeBERTaForNLI(XBERTForNLI):
    def __init__(
        self,
        label2id: Dict,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        max_model_len = 2048
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
        label2id = {
            "not_entailment": 0,
            "entailment": 1,
        }
        super().__init__(label2id, device, batch_size)


class RoBERTaForNLI(XBERTForNLI):
    def __init__(
        self,
        label2id: Dict,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        max_model_len = 512
        super().__init__(
            max_model_len, label2id=label2id, device=device, batch_size=batch_size
        )


class OurTrainedRoBERTa(RoBERTaForNLI):
    model_path = "./3nli_trained_roberta_large"

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
        label2id = {
            "not_entailment": 0,
            "entailment": 1,
        }
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
            self.max_model_len - 300
        )  # reserve some space (hard coded) for the claim to be checked
        self.model.to(device)
        self.model.eval()

    @override
    def score(self, doc: str, claim: str) -> float:
        def chunks(lst, n):
            current_chunk = []
            current_word_count = 0
            for sentence in lst:
                sentence_word_count = len(
                    self.tokenizer(
                        sentence,
                        padding=False,
                        add_special_tokens=False,
                        max_length=self.max_model_len,
                        truncation=True,
                    )["input_ids"]
                )
                if current_word_count + sentence_word_count > n:
                    yield " ".join(current_chunk)
                    current_chunk = [sentence]
                    current_word_count = sentence_word_count
                else:
                    current_chunk.append(sentence)
                    current_word_count += sentence_word_count
            if current_chunk:
                yield " ".join(current_chunk)

        doc_sents = sent_tokenize_with_newlines(doc)
        doc_sents = doc_sents or [""]
        doc_chunks = [
            chunk.replace(" \n ", "\n").strip()
            for chunk in chunks(doc_sents, self.default_chunk_size)
        ]
        doc_chunks = [chunk for chunk in doc_chunks if chunk != ""]
        claim_repeat = [claim] * len(doc_chunks)
        output = self._inference(doc_chunks, claim_repeat)
        return output

    def _inference(self, doc, claim):
        if isinstance(doc, str) and isinstance(claim, str):
            doc = [doc]
            claim = [claim]

        batch_input, _ = self._batch_tokenize(doc, claim)
        label_probs_list = []
        for mini_batch_input in batch_input:
            mini_batch_input = {
                k: v.to(self.device) for k, v in mini_batch_input.items()
            }
            with torch.no_grad():
                decoder_input_ids = torch.zeros(
                    (mini_batch_input["input_ids"].size(0), 1), dtype=torch.long
                ).to(self.device)
                outputs = self.model(
                    input_ids=mini_batch_input["input_ids"],
                    attention_mask=mini_batch_input["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                )
                logits = outputs.logits.squeeze(1)

                # 3 for no support and 209 for support
                label_logits = logits[:, torch.tensor([3, 209])].cpu()
                label_probs = torch.nn.functional.softmax(label_logits, dim=-1)
                label_probs_list.append(label_probs)
        label_probs = torch.cat(label_probs_list)
        max_support_prob = label_probs[:, 1].max().item()
        return max_support_prob

    def _batch_tokenize(self, doc, claim):
        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        original_text = [
            self.tokenizer.eos_token.join([one_doc, one_claim])
            for one_doc, one_claim in zip(doc, claim)
        ]
        batch_input = []
        batch_concat_text = []
        for mini_batch in chunks(original_text, self.batch_size):
            model_inputs = self.tokenizer(
                ["predict: " + text for text in mini_batch],
                max_length=self.max_model_len,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            batch_input.append(model_inputs)
            batch_concat_text.append(mini_batch)
        return batch_input, batch_concat_text
