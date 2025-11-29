# =============================================================
#  src/train.py — CTI-NER fine-tuning pipeline (softmax or CRF)
# =============================================================
"""
Usage
-----
# Softmax head (default)
python -m src.train

# CRF head
python -m src.train --head crf
"""
from __future__ import annotations

import os
import sys
import argparse
import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import evaluate
from datasets import DatasetDict

try:
    from torchcrf import CRF
except ImportError:
    CRF = None

from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)


# ---------------------------- Models ----------------------------
class RobertaSoftmaxNER(nn.Module):
    def __init__(self, backbone: AutoModel, num_labels: int):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(backbone.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        feats = self.backbone(input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(self.dropout(feats))
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, logits)


class RobertaCrfNER(nn.Module):
    def __init__(self, backbone: AutoModel, num_labels: int):
        super().__init__()
        if CRF is None:
            raise ImportError(
                "torchcrf is required for the CRF head. Install with "
                "`pip install torchcrf` or `pip install pytorch-crf`."
            )
        self.backbone = backbone
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(backbone.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        feats = self.backbone(input_ids, attention_mask=attention_mask).last_hidden_state
        emissions = self.classifier(self.dropout(feats))
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction="mean")
            return loss, emissions
        decoded = self.crf.decode(emissions, mask=attention_mask.bool())
        return None, decoded


# ----------------------------- CLI -----------------------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="CyberPeace-Institute/SecureBERT-NER",
                    help="HF model id or path")
    ap.add_argument("--head", choices=["softmax", "crf"], default="softmax")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    return ap.parse_args()


# --------------------------- Training ---------------------------
def _main() -> None:
    args = _parse_args()

    # Project root
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    from src.preprocess import load_aptner
    from src.utils import get_label_maps, align_labels_with_tokens

    # Logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("train")

    # Data
    raw: DatasetDict = load_aptner(os.path.join(proj_root, "data"))
    combined_tags = list(raw["train"]["ner_tags"]) + list(raw["validation"]["ner_tags"]) + list(raw["test"]["ner_tags"])
    label2id, id2label = get_label_maps(combined_tags)
    num_labels = len(label2id)
    O_ID = label2id["O"]

    logger.info(f"Loading tokenizer: {args.checkpoint}")
    tok = AutoTokenizer.from_pretrained(
        args.checkpoint, add_prefix_space=True, use_fast=True, trust_remote_code=True
    )

    def preprocess(batch: Dict[str, Any]) -> Dict[str, Any]:
        enc = tok(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=args.max_len,
            padding="max_length",
        )
        enc["labels"] = [
            align_labels_with_tokens(tags, enc.word_ids(i), label2id)
            for i, tags in enumerate(batch["ner_tags"])
        ]
        return enc

    ds = raw.map(preprocess, batched=True, remove_columns=["tokens", "ner_tags"])
    collator = DataCollatorForTokenClassification(tok)

    backbone = AutoModel.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = (RobertaCrfNER if args.head == "crf" else RobertaSoftmaxNER)(backbone, num_labels)

    # Metrics
    seq_f1 = evaluate.load("seqeval")

    def compute_metrics(out: Tuple[Any, Any]) -> Dict[str, float]:
        preds, labels = out
        if args.head != "crf":
            preds = preds.argmax(-1)
            preds = [
                [id2label[p] for p, l in zip(seq, lab) if l != -100]
                for seq, lab in zip(preds, labels)
            ]
        true = [[id2label[l] for l in lab if l != -100] for lab in labels]
        return {"f1": seq_f1.compute(predictions=preds, references=true)["overall_f1"]}

    # Custom Trainer (CRF mask fix)
    @dataclass
    class CRFTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **_):
            labels = inputs.pop("labels")
            if args.head == "crf":
                mask = labels != -100
                labels = labels.clone()
                labels[~mask] = O_ID
            loss, outputs = model(**inputs, labels=labels)
            return (loss, outputs) if return_outputs else loss

    trainer = (CRFTrainer if args.head == "crf" else Trainer)(
        model=model,
        args=TrainingArguments(
            output_dir=f"cti-ner-{args.head}",
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            num_train_epochs=args.epochs,
            learning_rate=args.lr or (1e-6 if args.head == "crf" else 4e-6),
            warmup_ratio=0.10,
            weight_decay=0.01,
            max_grad_norm=0.8,
            fp16=True,
            lr_scheduler_type="linear",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="none",
        ),
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(f"cti-ner-{args.head}-best")


# -------------------------------------------------------------
#  Entrypoint
# -------------------------------------------------------------
if __name__ == "__main__":
    _main()
