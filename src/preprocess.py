import os
from datasets import Dataset, DatasetDict
from .data_loader import read_conll


def load_aptner(data_dir: str) -> DatasetDict:
    """Convert APTNER CoNLL files into a Hugging Face DatasetDict."""
    file_map = {
        "train": "APTNERtrain.txt",
        "validation": "APTNERdev.txt",
        "test": "APTNERtest.txt",
    }
    data = {}
    for split, fname in file_map.items():
        path = os.path.join(data_dir, fname)
        sents, tags = read_conll(path)
        data[split] = {"tokens": sents, "ner_tags": tags}
    return DatasetDict({s: Dataset.from_dict(d) for s, d in data.items()})
