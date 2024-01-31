from eval import eval
from custom_tok import tokenizer
from segmentation import segmentation
import pandas as pd

def pipeline(path, tokeniz="tok", model='mod'):
    text = readfile(path)
    raw_toks = tokenizer(text, tokenizer=tokeniz)
    experts =  getExperts("../dataset_v20230110.tsv","../documents/dev/ROSENBLATT v. BAER_MCL.txt")
    print("experts", experts)
    experts = [tokenizer(sentence, tokenizer=tokeniz) for sentence in experts]
    print("tokens experts",experts)
    seg = segmentation(text, model=model)
    #retokenize
    print("seg : ",seg)
    #plus de tokenization
    #toks = [tokenizer(s, tokenizer=tokeniz) for s in seg]
    #print("toks : ",toks)
    return eval(experts, seg)


def readfile(path):
    text = open(path, "r")
    original = text.readlines()
    original = [ori[:-1] for ori in original]
    text = ""
    for line in original:
        text += line + " "
    return text 

def getExperts(pathExp, otherPath):
    title = otherPath.split("/")[-1].split('.txt')[0]
    df = pd.read_csv(pathExp, delimiter="	")
    content = df.loc[df["document"] == title]["content"]
    return list(content)

if __name__ == "__main__":
    # execute only if run as a script
    pipeline()