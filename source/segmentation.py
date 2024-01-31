#from tokenize import tokenizer
from custom_tok import tokenizer as custom_tok
import spacy
from nltk.tokenize import sent_tokenize
from spacy.lang.en import English


def segmentation(text, tokenizer=None, model=None):
    # tokenizer(text, model)
    
    match model : 
        case "spacy":
            #tokenization
            nlp = English()
            nlp.add_pipe("sentencizer")
            doc = nlp(text)
            #print("problem2", list(doc.sents))
            sentences = []
            for sentence in doc.sents:
                tokens = []
                for token in sentence : 
                    tokens += [token.text]
                sentences += [tokens]
            print("SENTENCES", sentences)
            #tokenization for return if not tokenized
            return sentences
        
        case "nltk":
            return sent_tokenize(text)#tokens
        
        case None: 
            return text