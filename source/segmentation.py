#from tokenize import tokenizer
import spacy
from nltk.tokenize import sent_tokenize

def segmentation(tokens, model='main'):
    # tokenizer(text, model)
    
    match model : 
        case "spacy":
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(tokens)
            #print("problem2", list(doc.sents))
            return [sent.text for sent in doc.sents]
        case "nltk":
            print(tokens)
            return sent_tokenize(tokens)