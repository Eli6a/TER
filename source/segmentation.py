#from tokenize import tokenizer
from custom_tok import tokenizer as custom_tok
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
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
                print('sentence text',sentence.text)
                sentences += [sentence.text]

                ##tokenization
                #tokens = []
                #for token in sentence : 
                #    tokens += [token.text]
                #sentences += [tokens]
                ##tokenization
            #tokenization for return if not tokenized
            #here save sentences
            return sentences
        
        case "nltk":
            """
            ## with tokenization
            if tokenizer is None :
                sentences = [custom_tok(sent, tokenizer="nltk-punkt")for sent in sent_tokenize(text)]
            else : 
                sentences = [custom_tok(sent, tokenizer="spacy")for sent in sent_tokenize(text)]
            ## with tokenization
            """
            #print("sentences nltk :", sentences)
            print("sent tokeniw",sent_tokenize(text))
            return sent_tokenize(text)#tokens
        
        case "naive":
            sentences = naive_segmentation(text)
            #print("sentences naive :", sentences)
            return sentences#tokens
        
        case None: 
            return text
        
def naive_segmentation(text):
    tokens = custom_tok(text, tokenizer="nltk-word")    
    sentences = []
    sentence = ""
    for token in tokens:
        if '.' in token:
            sentence += token
            sentences += [sentence]
            sentence = ""
        else:
            sentence += token    
            
    return sentences
