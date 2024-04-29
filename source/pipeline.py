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
    #experts = [tokenizer(sentence, tokenizer=tokeniz) for sentence in experts]
    #print("tokens experts",experts)
    #print("model", model)
    seg = segmentation(text, tokenizer=tokeniz, model=model)
    #print("Segmentation", seg)
    saveSegmentation(path, seg, model)
    save_differences(experts, seg, model, path)
    #print("seg : ",seg)
    
    # sklearn
    #print(eval_sklearn(experts, seg)) 
    
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
    df = pd.read_csv(pathExp, delimiter="	", quoting=csv.QUOTE_NONE, encoding='utf-8')
    content = df.loc[df["document"] == title]["content"]
    return list(content)

def evaluation(dir_path, tokeniz="tok", model='mod'):
    files = glob(dir_path)
    df = pd.DataFrame(columns=["model", "file", "precision", "recall", "F1_score"])
    for file in files :
        p, r, f = pipeline(file, tokeniz, model)
        if ((p + r + f) /3 )< 0.8 and model != 'naive':
            print(p,r,f, file)
        #pd.DataFrame({'precision':p,'recall':r, 'F1_score':f, 'file':file}, index=[0])
        df.loc[len(df)] = [model, file.split('/')[-1], p, r, f]
    df.to_csv("../stats/doc_train_stats_" + model + ".csv",index=False)
    return df["recall"].mean(), df["precision"].mean(), df["F1_score"].mean()

def saveSegmentation(path, seg, model):

    res = ""
    for i in range(len(seg)):
        temp_res = ""
            #for j in range(len(seg[i])):
            #    temp_res += seg[i][j] + ' '
            #res += temp_res
            #if i < len(seg)-1:
                    #print("seg i ", seg[i][j])
        res += seg[i] + "\n"
    with open("../outputs/"+model+"/"+path.split('/')[-1], 'w') as f:
          f.write(res)
  

def save_differences(ori, seg, model, title):
    path = ["expert", model]
    res = []
    precontext = 20
    postcontext = 20

    for j,doc in enumerate([ori, seg]):
        temp = ""
        for i in range(len(doc)):
            if i != len(doc) - 1:
                if len(doc[i]) < precontext:
                    precontext = len(doc[i])
                if len(doc[i+1]) < postcontext:
                    postcontext = len(doc[i+1])
                temp += doc[i][-precontext:] + ' | ' + doc[i+1][:postcontext] + '\n'
        with open("../errors/"+path[j]+"/"+title.split('/')[-1], 'w') as f:
            f.write(temp)



if __name__ == "__main__":
    # execute only if run as a script
    pipeline()