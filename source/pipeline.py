import segmentation, eval
from custom_tok import tokenizer


def pipeline(path, model='tok'):
    text = readfile(path)
    raw_toks = tokenizer(text, model)
    seg = segmentation(text)
    #eval(seg, raw_toks)


def readfile(path):
    text = open(path, "r")
    original = text.readlines()
    original = [ori[:-1] for ori in original]
    text = ""
    for line in original:
        text += line + " "
    return text 

if __name__ == "__main__":
    # execute only if run as a script
    pipeline()