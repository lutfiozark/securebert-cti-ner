# =============================================================
#  demo_gradio.py — CTI-NER interactive demo (RoBERTa + softmax)
# =============================================================
import os
from pathlib import Path

import torch
import torch.nn as nn
import gradio as gr
from transformers import AutoTokenizer, AutoModel, AutoConfig

# -------------------- Project paths ---------------------------
PROJ = Path(__file__).resolve().parent
CKPT = Path(os.environ.get("CTI_NER_CHECKPOINT", PROJ / "cti-ner-softmax-best"))

# -------------------- Model class (matches training) ----------
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


# -------------------- Label mapping ---------------------------
cfg = AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
cfg_id2label = getattr(cfg, "id2label", {}) or {}
cfg_id2label = {int(k): v for k, v in cfg_id2label.items()}

# Load trained weights first to know classifier size
state_dict = torch.load(CKPT / "pytorch_model.bin", map_location="cpu")
classifier_rows = state_dict["classifier.weight"].shape[0]

id2label = [cfg_id2label.get(i, f"LABEL_{i}") for i in range(classifier_rows)]
label2id = {lab: i for i, lab in enumerate(id2label)}
num_labels = classifier_rows

# -------------------- Tokenizer & Model -----------------------
tok = AutoTokenizer.from_pretrained(
    "CyberPeace-Institute/SecureBERT-NER",
    add_prefix_space=True,
    use_fast=True,
)

backbone = AutoModel.from_pretrained("CyberPeace-Institute/SecureBERT-NER")
model = RobertaSoftmaxNER(backbone, num_labels)

# Load trained weights
model.load_state_dict(state_dict)
model.eval()


# -------------------- Prediction function --------------------
def ner_predict(text: str):
    text = text.strip()
    if not text:
        return []

    enc = tok(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        add_special_tokens=True,
    )

    with torch.no_grad():
        logits = model(**enc)[1]  # (loss, logits) -> logits
        preds = logits.argmax(-1).squeeze().tolist()

    tokens = tok.convert_ids_to_tokens(enc["input_ids"].squeeze())
    out = []
    for tok_txt, lab_id in zip(tokens, preds):
        if tok_txt in tok.all_special_tokens:
            continue
        clean = tok_txt.replace("\u0120", " ").strip()  # strip RoBERTa space marker
        label = id2label[lab_id]
        out.append((clean, None if label == "O" else label))

    return out


# -------------------- Gradio UI -------------------------------
demo = gr.Interface(
    fn=ner_predict,
    inputs=gr.Textbox(lines=6, placeholder="Enter CTI text..."),
    outputs=gr.HighlightedText(label="Predicted entities"),
    title="CTI-NER Demo (Softmax)",
    description="Interactive inference with the fine-tuned softmax model.",
)

if __name__ == "__main__":
    demo.launch(share=True)
