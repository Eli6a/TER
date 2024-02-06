from eval import eval, eval_sklearn
from custom_tok import tokenizer
from segmentation import segmentation
import pandas as pd
from glob import glob
import csv

def pipeline(path, tokeniz="tok", model='mod'):
    text = readfile(path)
    #raw_toks = tokenizer(text, tokenizer=tokeniz)
    experts =  getExperts("../dataset_v20230110.tsv",path)
    #print("experts", experts)
    experts = [tokenizer(sentence.encode('latin1').decode('utf-8'), tokenizer=tokeniz) for sentence in experts]
    #print("tokens experts",experts)
    seg = segmentation(text, model=model)
    saveSegmentation(path, seg, model)
    print("seg : ",seg)
    
    # sklearn
    if (model == 'naive' or model == 'nltk'):
        print(eval_sklearn(experts, seg))
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
    df = pd.read_csv(pathExp, delimiter="	", quoting=csv.QUOTE_NONE, encoding='latin-1')
    content = df.loc[df["document"] == title]["content"]
    return list(content)

def evaluation(dir_path, tokeniz="tok", model='mod'):
    files = glob(dir_path)
    df = pd.DataFrame(columns=["model", "file", "precision", "recall", "F1_score"])
    for file in files :
        p, r, f = pipeline(file, tokeniz, model)
        #pd.DataFrame({'precision':p,'recall':r, 'F1_score':f, 'file':file}, index=[0])
        df.loc[len(df)] = [model, file.split('/')[-1], p, r, f]
    df.to_csv("../stats/doc_train_stats.csv",index=False)
    return df["recall"].mean(), df["precision"].mean(), df["F1_score"].mean()

def saveSegmentation(path, seg, model):
    res = ""
    for i in range(len(seg)):
        temp_res = ""
        for j in range(len(seg[i])):
            temp_res += seg[i][j] + ' '
        res += temp_res
        if i < len(seg)-1:
                #print("seg i ", seg[i][j])
            res += "\n"

    with open("../outputs/"+model+"/"+path.split('/')[-1], 'w') as f:
        f.write(res)
if __name__ == "__main__":
    # execute only if run as a script
    pipeline()