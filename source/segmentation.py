import os

def segmentation(title, text, model='main'):
    # tokenizer(text, model)
    
    if model == 'spacy':
        return segmentation_spacy(title, text)
    
    if model == 'nltk':
        return segmentation_nltk(title, text)
    
def segmentation_spacy(title, text):
    import spacy

    script_directory = os.path.dirname(__file__)
    output_file_path = os.path.join(script_directory, '../outputs/spacy/' + title + '.txt')

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    try:
        with open(output_file_path, 'w') as f:
            for sent in doc.sents:
                f.write(sent.text + '\n')
                
        return output_file_path
        
    except Exception as e:
        print(f"Erreur lors de l'écriture du fichier : {e}")
        return None

def segmentation_nltk(title, text):
    from nltk.tokenize import sent_tokenize

    script_directory = os.path.dirname(__file__)
    output_file_path = os.path.join(script_directory, '../outputs/nltk/' + title + '.txt')
        
    try:
        with open(output_file_path, 'w') as f:
            for sent in sent_tokenize(text):
                f.write(sent + '\n')
        return output_file_path
    
    except Exception as e:
        print(f"Erreur lors de l'écriture du fichier : {e}")
        return None
 