from typing import List, Dict, Any, Tuple
from seqeval.metrics import f1_score


def get_label_maps(tag_seqs: List[List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = sorted({lab for seq in tag_seqs for lab in seq})
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def align_labels_with_tokens(
    labels: List[str], word_ids: List[Any], label2id: Dict[str, int]
) -> List[int]:
    """
    Align word-level labels to tokenized ids.
    - None or out-of-range indices -> -100 (ignored by loss)
    - Unknown labels -> "O"
    """
    O_id = label2id.get("O", 0)
    aligned, prev = [], None
    for idx in word_ids:
        if idx is None or idx >= len(labels):
            aligned.append(-100)
        elif idx != prev:
            aligned.append(label2id.get(labels[idx], O_id))
        else:
            aligned.append(label2id.get(labels[idx], O_id))
        prev = idx
    return aligned


def f1_metric(eval_pred, label_list: List[str]):
    preds, labels = eval_pred
    preds = preds.argmax(-1)
    true_p, true_l = [], []
    for p, l in zip(preds, labels):
        true_p.append([label_list[x] for x, y in zip(p, l) if y != -100])
        true_l.append([label_list[y] for x, y in zip(p, l) if y != -100])
    return {"f1": f1_score(true_l, true_p)}
