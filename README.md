# CTI Named Entity Recognition (APTNER)

Fine-tuning and evaluating NER models (RoBERTa/CRF or softmax head) on the APTNER cyber threat intelligence dataset, plus a lightweight Gradio demo.

## Quickstart
- Prerequisites: Python 3.10+, `pip install -r requirements.txt`.
- Data: place `APTNERtrain.txt`, `APTNERdev.txt`, `APTNERtest.txt` under `data/`. The dataset comes from the APTNER paper (see citation below); verify redistribution terms before committing it.
- Train (softmax default): `python -m src.train`
- Train with CRF head: `python -m src.train --head crf`
- Evaluate a checkpoint: `python -m src.evaluate --checkpoint cti-ner-softmax-best --split test`
- Gradio demo (expects a trained softmax checkpoint in `cti-ner-softmax-best/`): `python demo_gradio.py`
- Optional DAPT pretraining on CTI tweets: `python run_dapt.py`

## Notes on sharing to GitHub
- Model weights and optimizer states are large; they are `.gitignore`d. Use Git LFS if you plan to version checkpoints.
- `data/`, `.gradio/`, `eval_tmp/`, `__pycache__/`, and other generated artifacts are ignored to keep the repo clean.
- Check `requirements.txt` against your environment before publishing.

## Repository layout
- `src/train.py` — fine-tuning loop (softmax or CRF head)
- `src/evaluate.py` — offline evaluation for a saved checkpoint
- `src/preprocess.py`, `src/data_loader.py`, `src/utils.py` — data loading and label utilities
- `demo_gradio.py` — interactive tagging demo using the softmax head
- `run_dapt.py` — example domain-adaptive pretraining script
- `cti-ner-softmax*` — saved checkpoints (ignored by git)

## Dataset citation
APTNER: Xuren Wang, Songheng He, Zihan Xiong, Xinxin Wei, Zhangwei Jiang, Sihan Chen, Jun Jiang. “APTNER: A Specific Dataset for NER Missions in Cyber Threat Intelligence Field.” CSCWD 2022.
