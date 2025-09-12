from abc import ABC, abstractmethod
from typing import List, override

import torch
from nltk.tokenize import sent_tokenize
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# wrapper class for model to be evaluated
class Scorer(ABC):
    @abstractmethod
    def score(self, doc: str, claim: str) -> float:
        pass


class RoBERTaForNLI(Scorer):
    def __init__(
        self,
        model_path_or_name: str,
        max_sentences: int,
        device: str = "cpu",
        batch_size: int = 16,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path_or_name
        )
        self.positive_label = 0
        self.max_sentences = max_sentences
        self.device = device
        self.batch_size = batch_size
        self.model.to(device)
        self.model.eval()

    @override
    def score(self, doc: str, claim: str) -> float:
        subdocs = self._window_doc_by_sentences(doc)
        claims = [claim] * len(subdocs)

        support_probs = []
        for i in range(0, len(subdocs), self.batch_size):
            batched_subdocs = subdocs[i : i + self.batch_size]
            batched_claims = claims[i : i + self.batch_size]
            batched_tokenized = self.tokenizer(
                text=batched_subdocs,
                text_pair=batched_claims,
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                padding=True,
                return_token_type_ids=True,
                return_tensors="pt",
            )
            input_ids = batched_tokenized["input_ids"].to(self.device)
            attention_mask = batched_tokenized["attention_mask"].to(self.device)
            token_type_ids = batched_tokenized["token_type_ids"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask, token_type_ids)
            probs = torch.softmax(outputs.logits, dim=1)
            support_probs += probs[:, self.positive_label].tolist()

        return max(support_probs) if support_probs else 0.0

    def _window_doc_by_sentences(self, doc: str) -> List[str]:
        doc_sentences = sent_tokenize(doc)
        subdocs = []
        for window_size in range(1, self.max_sentences + 1):
            for i in range(0, len(doc_sentences) - window_size + 1):
                subdoc = " ".join(doc_sentences[i : i + window_size])
                subdocs.append(subdoc)
        return subdocs


# Adapt code from https://github.com/Liyan06/MiniCheck/blob/main/minicheck/inference.py
class MiniCheck(Scorer):
    def __init__(
        self, model_path_or_name: str, device: str = "cpu", batch_size: int = 16
    ):
        if model_path_or_name == "lytang/MiniCheck-Flan-T5-Large":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path_or_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)

            self.max_model_len = 2048
            self.max_output_length = 256

        else:
            if model_path_or_name == "lytang/MiniCheck-RoBERTa-Large":
                self.max_model_len = 512
            elif model_path_or_name == "lytang/MiniCheck-DeBERTa-v3-Large":
                self.max_model_len = 2048

            else:
                raise ValueError(f"Unsupported model: {model_path_or_name}")

            config = AutoConfig.from_pretrained(
                model_path_or_name,
                num_labels=2,
                finetuning_task="text-classification",
                revision="main",
                token=None,
            )
            config.problem_type = "single_label_classification"

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path_or_name, use_fast=True, revision="main", token=None
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path_or_name,
                config=config,
                revision="main",
                token=None,
                ignore_mismatched_sizes=False,
            )

        self.model_path_or_name = model_path_or_name
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
            if self.model_path_or_name == "lytang/MiniCheck-Flan-T5-Large":
                current_chunk = []
                current_word_count = 0
                for sentence in lst:
                    sentence_word_count = len(sentence.split())
                    if current_word_count + sentence_word_count > n:
                        yield " ".join(current_chunk)
                        current_chunk = [sentence]
                        current_word_count = sentence_word_count
                    else:
                        current_chunk.append(sentence)
                        current_word_count += sentence_word_count
                if current_chunk:
                    yield " ".join(current_chunk)
            else:
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

        def sent_tokenize_with_newlines(text):
            blocks = text.split("\n")
            tokenized_blocks = [sent_tokenize(block) for block in blocks]
            tokenized_text = []
            for block in tokenized_blocks:
                tokenized_text.extend(block)
                tokenized_text.append("\n")

            return tokenized_text[:-1]

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
                if self.model_path_or_name == "lytang/MiniCheck-Flan-T5-Large":
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
                else:
                    outputs = self.model(**mini_batch_input)
                    logits = outputs.logits
                    label_probs = torch.nn.functional.softmax(logits, dim=1)
                label_probs_list.append(label_probs)
        label_probs = torch.cat(label_probs_list)
        max_support_prob = label_probs[:, 1].max().item()
        return max_support_prob

    def _batch_tokenize(self, doc, claim):
        assert isinstance(doc, list) and isinstance(claim, list)
        assert len(doc) == len(claim), "doc and claim should be in the same length."

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
            if self.model_path_or_name == "lytang/MiniCheck-Flan-T5-Large":
                model_inputs = self.tokenizer(
                    ["predict: " + text for text in mini_batch],
                    max_length=self.max_model_len,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
            else:
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
