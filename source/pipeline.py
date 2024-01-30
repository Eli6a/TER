import segmentation, eval
from custom_tok import tokenizer


def pipeline(path, model='tok'):
    text = open(path, "r")
    original = text.readlines()
    original = [ori[:-1] for ori in original]
    raw_toks = tokenizer(original)
    print(raw_toks)
    #seg = segmentation(text)
    #eval(seg, raw_toks)


if __name__ == "__main__":
    # execute only if run as a script
    pipeline()