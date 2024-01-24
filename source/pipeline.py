import segmentation, eval
from tokenize import tokenizer


def pipeline(path):
    text = file(path)
    raw_toks = tokenizer(text)
    seg = segmentation(text)
    eval(seg, raw_toks)


if __name__ == "__main__":
    # execute only if run as a script
    pipeline()