import segmentation, eval
#from tokenize import tokenizer


def pipeline(path='C:\\Users\\elisa\\Downloads\\TER\\documents\\train\\ROSENBLATT v. BAER_MCL.txt', model='spacy'):
    text = ''
    
    with open(path, 'r') as fichier:
        text = fichier.read()
    
   # raw_toks = tokenizer(text)
    seg = segmentation.segmentation('ROSENBLATT v. BAER_MCL', text, model='spacy')
    seg2 = segmentation.segmentation('ROSENBLATT v. BAER_MCL', text, model='nltk')
   # eval(seg, raw_toks)


if __name__ == "__main__":
    # execute only if run as a script
    pipeline()