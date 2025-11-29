# =============================================================
#  src/evaluate.py — CTI-NER evaluation script
# =============================================================
"""
Usage
-----
python -m src.evaluate --checkpoint /path/to/model_dir --split test
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, List

import torch
import numpy as np
import evaluate
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    DataCollatorForTokenClassification,
)

# --------------------------- Project paths --------------------
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from src.preprocess import load_aptner
from src.utils import get_label_maps, align_labels_with_tokens
from src.train import RobertaSoftmaxNER, RobertaCrfNER

# --------------------------- CLI ------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True, help="Trained model directory or HF id")
ap.add_argument("--split", default="test", choices=["validation", "test"])
ap.add_argument("--max_len", type=int, default=512)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# --------------------------- Data -----------------------------
raw: DatasetDict = load_aptner(os.path.join(proj_root, "data"))
combined_tags = list(raw["train"]["ner_tags"]) + list(raw["validation"]["ner_tags"]) + list(raw["test"]["ner_tags"])
label2id, id2label = get_label_maps(combined_tags)
num_labels = len(label2id)

tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, add_prefix_space=True)


def preprocess(batch: Dict[str, Any]) -> Dict[str, Any]:
    tok = tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=args.max_len,
        padding="max_length",
    )
    tok["labels"] = [
        align_labels_with_tokens(tags, tok.word_ids(i), label2id)
        for i, tags in enumerate(batch["ner_tags"])
    ]
    return tok


eval_ds = raw[args.split].map(preprocess, batched=True, remove_columns=["tokens", "ner_tags"])
data_collator = DataCollatorForTokenClassification(tokenizer)

# --------------------------- Model ----------------------------
cfg_path = os.path.join(args.checkpoint, "config.json")
if os.path.exists(cfg_path):
    with open(cfg_path) as cf:
        arch = json.load(cf).get("architectures", ["RobertaForTokenClassification"])[0].lower()
else:  # remote id or missing config
    arch = AutoConfig.from_pretrained(args.checkpoint).architectures[0].lower()

head_type = "crf" if "crf" in arch else "softmax"
logging.info(f"Loading model ({head_type}) from {args.checkpoint}")

# backbone + wrapper
backbone = AutoModel.from_pretrained(args.checkpoint, trust_remote_code=True)
model = RobertaCrfNER(backbone, num_labels) if head_type == "crf" else RobertaSoftmaxNER(backbone, num_labels)

# Weight file: prefer .bin; if missing but .safetensors exists, convert once
bin_path = os.path.join(args.checkpoint, "pytorch_model.bin")
safe_path = os.path.join(args.checkpoint, "model.safetensors")

if not os.path.exists(bin_path) and os.path.exists(safe_path):
    from safetensors.torch import load_file
    torch.save(load_file(safe_path), bin_path)

if os.path.exists(bin_path):
    state = torch.load(bin_path, map_location="cpu")
    model.load_state_dict(state, strict=False)

model.eval().to(args.device)

# --------------------------- Metrics --------------------------
seq_f1 = evaluate.load("seqeval")

ids, preds = [], []
for i in range(0, len(eval_ds), args.batch):
    batch = eval_ds[i : i + args.batch]
    ids.append(batch["labels"])
    with torch.no_grad():
        out = model(
            input_ids=torch.tensor(batch["input_ids"]).to(args.device),
            attention_mask=torch.tensor(batch["attention_mask"]).to(args.device),
        )
    logits = out[1]
    if head_type == "softmax":
        preds.append(torch.argmax(logits, dim=-1).cpu().tolist())
    else:  # CRF already returns decoded labels
        preds.append(logits)

# ID -> label, skip padding
flat_preds, flat_true = [], []
for p_batch, t_batch in zip(preds, ids):
    for p_seq, t_seq in zip(p_batch, t_batch):
        t_tags, p_tags = [], []
        for p_id, t_id in zip(p_seq, t_seq):
            if t_id == -100:
                continue
            t_tags.append(id2label[t_id])
            p_tags.append(id2label[p_id])
        flat_true.append(t_tags)
        flat_preds.append(p_tags)

metrics = seq_f1.compute(predictions=flat_preds, references=flat_true)


def _to_py(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(x) for x in obj]
    return obj


logging.info(json.dumps(_to_py(metrics), indent=2, ensure_ascii=False))
