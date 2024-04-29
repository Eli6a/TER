#from tokenize import tokenizer
from custom_tok import tokenizer as custom_tok
import spacy
from spacy.language import Language
from nltk.tokenize import sent_tokenize, word_tokenize
from spacy.lang.en import English
# from ersatz.split import split, parse_args, EvalModel
# from ersatz.utils import get_model_path, list_models, MODELS
# from ersatz.candidates import PunctuationSpace
import argparse
from io import StringIO
import re
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
        
        case "custom_spacy":
            # try:
            #     nlp = spacy.load("../models/custom_spacy_model")
            # except OSError:
            #     
            nlp = custom_spacy_model()            
            
            doc = nlp(text)
            sentences = []
            for sentence in doc.sents:
                sentences += [sentence.text]
            return sentences
        
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
            sentence += token + ' '
            
    return sentences

# Définir une fonction pour vérifier si un token est un chiffre romain
def is_roman_numeral(token):
    return bool(re.match(r'^(?:i[vx]|v[li]*|x[vli]*)$', token.text.lower()))

# Ajouter une règle de segmentation personnalisée
def custom_segmentation(doc):
    to_end_after_bracket = False
    for i, token in enumerate(doc[:-1]):
        
        # si mot qui créer une phrase
        if token.text == "Id." or token.text == "See":
            token.is_sent_start = True
        
        if token.text == "Ibid." :
            token.is_sent_start = True
            if i < len(doc) - 2 :
                doc[i + 1].is_sent_start = True

        
        # si point
        if token.text == ".":
            token.is_sent_start = False               
            
            # Les parenthèses / crochets font partie de la phrase précédente
            if doc[token.i + 1].text == "[" or doc[token.i + 1].text == "(" or doc[token.i + 2].text == "[" or doc[token.i + 2].text == "(":
                doc[token.i + 1].is_sent_start = False
                to_end_after_bracket = True
            # Le point est suivi d'une minuscule   
            elif re.match(r'^[a-z]', doc[i + 1].text):
                doc[i + 1].is_sent_start = False
            
            # si ressemble au point d'un sigle
            elif (i > 0 and re.match(r'^[A-Z][A-Za-z]{0,3}$|^[a-z]{1,2}$', doc[i-1].text) and i < len(doc) - 2):
                doc[i + 1].is_sent_start = False
            
            # si dernier point des points de suspension    
            elif (i > 2 and doc[i - 1].text == "." and doc[i - 2].text == ".")  and i < len(doc) - 2:
                doc[i + 1].is_sent_start = False
                
            continue
        
        # si fermer parenthèse ou crochet et to_end_after_bracket est vrai
        if (token.text == "]" or token.text == ")") and to_end_after_bracket:
            # si suivi de parenthèses ou crochets, la phrase continue, sinon fin de phrase
            if (doc[i + 1].text != "(" or doc[i + 1].text != "["):
                doc[i + 1].is_sent_start = True
                to_end_after_bracket = False
            # si suivi ponctuation, la phrase continue
            elif (doc[i+1].text == "." or doc[i+1].text == ";" or token.text == ":"):
                to_end_after_bracket = False
            continue
        
        # si est entre parenthèse, continue la phrase    
        if to_end_after_bracket :
            doc[i + 1].is_sent_start = False
            continue
        
        # si virgule, la virgule fait partie de la phrase précédente et pas de fin de phrase
        if token.text == "," and i < len(doc) - 2 :
            token.is_sent_start = False
            doc[i + 1].is_sent_start = False
            continue
            
        # si ponctuation, la ponctuation fait partie de la phrase précédente
        if token.text == ";" or token.text == ":":
            token.is_sent_start = False
            # si suivie d'une minuscule, ne marque pas la fin de la phrase, sinon oui
            if (i < len(doc) - 2 and re.match(r'^[a-z]', doc[i + 1].text)):
                doc[i + 1].is_sent_start = False
            elif (i < len(doc) - 2 and doc[i + 1].text == "\""):
                doc[i + 1].is_sent_start = True
                if i < len(doc) - 3:
                    doc[i + 2].is_sent_start = False
            else :
                doc[i + 1].is_sent_start = True
            continue
    return doc

# Ajouter la fonction de segmentation personnalisée au pipeline spaCy
@Language.component("custom_segmentation")
def custom_segmentation_component(doc):
    return custom_segmentation(doc)

def custom_spacy_model():
    # Charger le modèle spaCy de base
    nlp = spacy.load("en_core_web_sm")

    # Insérer la segmentation personnalisée au début du pipeline
    nlp.add_pipe("custom_segmentation", before="tagger")

    nlp.to_disk("../models/custom_spacy_model")
    
    return nlp

