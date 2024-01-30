from nltk.tokenize import wordpunct_tokenize, word_tokenize
from spacy.tokenizer import Tokenizer as spacyTok
from spacy.lang.en import English
def tokenizer_openNlp():
    return 0

def tokenizer(text, tokenizer='main'):
    tokens = []
    match tokenizer :
        case "nltk-punkt" :
            tokens = wordpunct_tokenize(text)
        case "nltk-word":
            tokens = word_tokenize(text)
        case "spacy":
            nlp = English()
            tokenizer = spacyTok(nlp)
            tokens = tokenizer(text)
        case None:
            tokens = text
    return tokens

