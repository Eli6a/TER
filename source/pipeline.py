import segmentation, eval
import os
from custom_tok import tokenizer


def pipeline(path='', model='spacy'):
    
    default_title = 'ROSENBLATT v. BAER_MCL'
    if path == '':
        script_directory = os.path.dirname(__file__)
        path = os.path.join(script_directory, '../documents/train/' + default_title + '.txt')
    
    text = readfile(path)
    
   # raw_toks = tokenizer(text, model)
    seg = segmentation.segmentation(default_title, text, model='spacy')
    #seg2 = segmentation.segmentation(default_title, text, model='nltk')
   # eval(seg, raw_toks)


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