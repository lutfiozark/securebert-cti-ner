# CTI Named Entity Recognition (APTNER)

RoBERTa-based NER that detects **21 STIX 2.1–aligned cyber threat intelligence (CTI) entity types**. Trained on the APTNER dataset with a **RoBERTa + BiGRU + CRF** architecture; a softmax head is also supported. A Gradio demo provides interactive tagging with a trained checkpoint.

---

## 1) Overview
- **Model:** RoBERTa-base + BiGRU + CRF (optional softmax head)
- **Domain:** Cyber Threat Intelligence (CTI)
- **Entities:** 21 STIX 2.1 labels
- **Performance (APTNER test):** micro F1 ≈ 0.96, macro F1 ≈ 0.93
- **Demo:** Gradio UI with a trained softmax checkpoint

## 2) Quickstart
- Prereqs: Python 3.10+, `pip install -r requirements.txt`
- Data: place `APTNERtrain.txt`, `APTNERdev.txt`, `APTNERtest.txt` under `data/` (check license/redistribution; not committed)
- Train (softmax default): `python -m src.train`
- CRF head: `python -m src.train --head crf`
- Evaluate: `python -m src.evaluate --checkpoint cti-ner-softmax-best --split test`
- Demo: `python demo_gradio.py` (expects checkpoint at `cti-ner-softmax-best/`; override with env `CTI_NER_CHECKPOINT`)
- Optional DAPT: `python run_dapt.py`

## 3) APTNER Dataset
APTNER is built from CTI reports and labeled in BIOES format with 21 entity types.

### 3.1 Statistics
| Field            | Value    |
| ---------------- | -------- |
| Sentences        | 10,984   |
| Tokens           | 260,134  |
| Entities         | 39,565   |
| Entity types     | 21       |
| Label format     | BIOES    |
| Language         | English  |
| Domain           | CTI      |

### 3.2 Sample labels
| Label  | Description                     | Example              |
| ------ | --------------------------------| -------------------- |
| APT    | Advanced threat group           | APT28                |
| SECTEAM| Security team/org               | research team        |
| VULID  | Vulnerability ID                | CVE-2021-26855       |
| VULNAME| Vulnerability name              | Heartbleed           |
| MAL    | Malware name                    | WannaCry             |
| TOOL   | Attack / hacking tool           | Mimikatz             |
| IP     | IP address                      | 192.168.0.1          |
| DOM    | Domain                          | malicious.com        |
| URL    | URL                             | http://example.com   |
| EMAIL  | Email                           | user@example.com     |
| HASH   | MD5/SHA1/SHA2                   | e99a18…              |
| ACT    | Attack technique                | spear phishing       |
| IDTY   | Identity / credential           | username             |
| OS     | Operating system                | Windows 10           |
| PROT   | Network protocol                | HTTP                 |

## 4) Architecture & Training
- **Flow:** RoBERTa-base → BiGRU → CRF (or softmax)
- **Tokenizer:** BPE with BIOES-aligned subword labels
- **Loss:** CRF negative log-likelihood; softmax cross-entropy
- **Trainer:** Hugging Face Trainer; early stopping on F1

### 4.1 Hyperparameters
| Param                 | Value                  |
| --------------------- | ---------------------- |
| Optimizer             | AdamW                  |
| Learning rate         | 5e-5 (CRF often 1e-6) |
| Epochs                | ≤10 (typically stops ~4–5) |
| Batch size            | 32                     |
| Dropout               | 0.1                    |
| Weight decay          | 0.01                   |
| Max seq length        | 256 (eval 512)         |
| Early stopping        | F1-based               |

## 5) Results (APTNER test)

### 5.1 Per-class P/R/F1
| Label  | P    | R    | F1   |
| ------ | ---- | ---- | ---- |
| APT    | 0.90 | 0.88 | 0.89 |
| SECTEAM| 0.92 | 0.89 | 0.90 |
| LOC    | 0.95 | 0.94 | 0.94 |
| TIME   | 0.93 | 0.92 | 0.92 |
| VULNAME| 0.88 | 0.86 | 0.87 |
| VULID  | 0.99 | 0.99 | 0.99 |
| TOOL   | 0.91 | 0.92 | 0.92 |
| MAL    | 0.90 | 0.91 | 0.90 |
| FILE   | 0.94 | 0.93 | 0.93 |
| MD5    | 0.99 | 0.98 | 0.98 |
| SHA1   | 0.98 | 0.99 | 0.99 |
| SHA2   | 0.99 | 0.99 | 0.99 |
| IDTY   | 0.85 | 0.84 | 0.85 |
| ACT    | 0.81 | 0.79 | 0.80 |
| DOM    | 0.96 | 0.97 | 0.96 |
| ENCR   | 0.95 | 0.93 | 0.94 |
| EMAIL  | 0.97 | 0.98 | 0.97 |
| OS     | 0.96 | 0.95 | 0.95 |
| PROT   | 0.98 | 0.97 | 0.98 |
| URL    | 0.96 | 0.95 | 0.95 |
| IP     | 0.99 | 0.99 | 0.99 |

### 5.2 Summary metrics
| Average | Precision | Recall | F1   |
| ------- | --------- | ------ | ---- |
| Micro   | 0.96      | 0.95   | 0.96 |
| Macro   | 0.93      | 0.92   | 0.93 |

**CRF effect:** ≈ +2 F1 versus a softmax-only head due to sequence-level consistency.

## 6) Repository Layout
- `src/train.py` — training loop (softmax or CRF)
- `src/evaluate.py` — evaluate a saved checkpoint
- `src/preprocess.py`, `src/data_loader.py`, `src/utils.py` — data loading and label alignment
- `demo_gradio.py` — Gradio demo with the trained softmax model
- `run_dapt.py` — domain-adaptive pretraining on CTI tweets
- `cti-ner-softmax*` — checkpoints (ignored by git)

## 7) Sharing Notes
- Large model/optimizer files are ignored; use Git LFS if you need to version checkpoints.
- `data/`, `.gradio/`, `eval_tmp/`, `__pycache__/`, and other generated artifacts are in `.gitignore`.
- Verify APTNER licensing/redistribution before sharing the dataset.

## 8) Citation
APTNER: Xuren Wang, Songheng He, Zihan Xiong, Xinxin Wei, Zhangwei Jiang, Sihan Chen, Jun Jiang. “APTNER: A Specific Dataset for NER Missions in Cyber Threat Intelligence Field.” CSCWD 2022.
