import segmentation, eval
import os

#from tokenize import tokenizer


def pipeline(path='ROSENBLATT v. BAER_MCL', model='spacy'):
    text = ''
    script_directory = os.path.dirname(__file__)
    input_file_path = os.path.join(script_directory, '../documents/train/' + path + '.txt')
    with open(input_file_path, 'r') as fichier:
        text = fichier.read()
    
   # raw_toks = tokenizer(text)
    seg = segmentation.segmentation('ROSENBLATT v. BAER_MCL', text, model='spacy')
    seg2 = segmentation.segmentation('ROSENBLATT v. BAER_MCL', text, model='nltk')
   # eval(seg, raw_toks)


if __name__ == "__main__":
    # execute only if run as a script
    pipeline()