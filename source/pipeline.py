import segmentation, eval
import os
from custom_tok import tokenizer


def pipeline(path='ROSENBLATT v. BAER_MCL', model='spacy'):
    script_directory = os.path.dirname(__file__)
    input_file_path = os.path.join(script_directory, '../documents/train/' + path + '.txt')
    text = readfile(input_file_path)

    
   # raw_toks = tokenizer(text, model)
    seg = segmentation.segmentation('ROSENBLATT v. BAER_MCL', text, model='spacy')
    #seg2 = segmentation.segmentation('ROSENBLATT v. BAER_MCL', text, model='nltk')
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