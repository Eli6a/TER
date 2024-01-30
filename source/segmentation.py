#from tokenize import tokenizer
import spacy
from nltk.tokenize import sent_tokenize

def segmentation(tokens, model='main'):
    # tokenizer(text, model)
    
    match model : 
        case "spacy":
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(tokens)
            return doc.sents
        case "nltk":
            print(tokens)
            return sent_tokenize(tokens)