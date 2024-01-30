from sklearn.metrics import f1_score, recall_score, precision_score

def eval(original, segmented):
    print(original)
    print(segmented)
    idx_ori = indexes_from_sentences(original)
    idx_segmented = indexes_from_sentences(segmented)
    f1 = f1_score(idx_ori, idx_segmented) 
    recall = recall_score(idx_ori, idx_segmented)
    precision = precision_score(idx_ori, idx_segmented)
    print("Precision : ", precision,"\nRecall : ", recall,"\nF1 : ", f1)
    return precision, recall, f1

    

def indexes_from_sentences(sentences):
    indices = []
    offset = 0
    for sentence in sentences[:-1]:
        offset += len(sentence) + 1
        indices += [offset]
    return indices

def evaluate_indices(ori, seg):
    true, pred = set(true), set(pred)
    tp = len(pred.intersection(true))
    fp = len(pred - true)
    fn = len(true - pred)

    return tp, fp, fn

def score(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2*(precision*recall)/(precision+recall) if precision + recall != 0 else 0
    return precision, recall, f1