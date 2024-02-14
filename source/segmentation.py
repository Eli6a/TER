#from tokenize import tokenizer
from custom_tok import tokenizer as custom_tok
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from spacy.lang.en import English
from ersatz.split import split, parse_args, EvalModel
from ersatz.utils import get_model_path, list_models, MODELS
from ersatz.candidates import PunctuationSpace
import argparse
from io import StringIO
import torch

def segmentation(text, tokenizer=None, model=None, args=None):
    # tokenizer(text, model)
    
    match model : 

        case "ersatz":
            input_file = text.split('\n')
            
            output_file = StringIO()
            if torch.cuda.is_available() and not args.cpu:
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

            model_path = get_model_path("en")
            model = EvalModel(model_path)

            model.model = model.model.to(device)
            model.device = device
            with torch.no_grad():
                output_file = model.split(input_file, output_file, 16, candidates=PunctuationSpace())
            return output_file.getvalue().strip().split('\n')
        
        case "spacy":
            #tokenization
            nlp = English()
            nlp.add_pipe("sentencizer")
            doc = nlp(text)
            #print("problem2", list(doc.sents))
            sentences = []
            for sentence in doc.sents:
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
