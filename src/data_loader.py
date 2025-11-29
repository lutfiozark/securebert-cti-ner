from typing import List, Tuple

# Skip noisy single-token URL/IP labels if desired
SKIP_TAGS = {"I-IP", "B-IP", "I-URL", "B-URL"}


def read_conll(filepath: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Reads a CoNLL-style file and returns (sentences, tags).
    - Empty/whitespace lines mark sentence boundaries.
    - Lines with <2 columns are skipped.
    - Tags in SKIP_TAGS are skipped.
    """
    sents, labs, tokens, tags = [], [], [], []
    with open(filepath, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                if tokens:
                    sents.append(tokens)
                    labs.append(tags)
                    tokens, tags = [], []
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            word, tag = parts[0], parts[-1]
            if tag in SKIP_TAGS:
                continue
            tokens.append(word)
            tags.append(tag)
        if tokens:
            sents.append(tokens)
            labs.append(tags)
    return sents, labs
