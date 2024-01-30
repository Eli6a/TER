import eval
from custom_tok import tokenizer
from segmentation import segmentation

def pipeline(path, tokeniz="tok", model='mod'):
    text = readfile(path)
    raw_toks = tokenizer(text, tokenizer=tokeniz)
    seg = segmentation(raw_toks, model=model)
    print("seg : ",seg)
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