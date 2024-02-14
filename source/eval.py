from sklearn.metrics import f1_score, recall_score, precision_score
import re
def eval(original, segmented):
    idx_ori = indexes_from_sentences(original)
    idx_segmented = indexes_from_sentences(segmented)
    #print("ori", original)
    #print(idx_ori)
    #print("seg", segmented)
    #print(idx_segmented)
    precision, recall, f1 = evaluate_indices(idx_ori, idx_segmented)
    #print("Precision : ", precision,"\nRecall : ", recall,"\nF1 : ", f1)
    return precision, recall, f1


def indexes_from_sentences(sentences):
    indices = []
    offset = 0
    for sentence in sentences:
        offset += len(re.sub('\s*','',sentence))
        indices += [offset]
    return indices

def evaluate_indices(ori, seg):
    true, pred = set(ori), set(seg)
    tp = len(pred.intersection(true))
    fp = len(pred - true)
    fn = len(true - pred)
    return score(tp, fp, fn)

def score(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2*(precision*recall)/(precision+recall) if precision + recall != 0 else 0
    return precision, recall, f1

def tokens_to_0_1(tokens):
    tot_array = [0] * sum(len(array) for array in tokens)

    start_index = 0
    for array in tokens:
        end_index = start_index + len(array)
        tot_array[end_index - 1] = 1
        start_index = end_index
    
    return tot_array

def eval_sklearn(expert, segmented):
    print("expert",len(expert))
    print("segmented", len(segmented))
    """
    for seg in segmented : 
        print("lens : ", len(seg))
    for seg in expert : 
        print("exp : ", len(seg))
    """
    expert = tokens_to_0_1(expert)
    segmented = tokens_to_0_1(segmented)
    print("expert",len(expert), expert)
    print("segmented", len(segmented), segmented)
    precision = precision_score(expert, segmented)
    recall = recall_score(expert, segmented)
    f1 = f1_score(expert, segmented)
    return precision, recall, f1

def index_differences(array1, array2):
    differences = [i for i, (a, b) in enumerate(zip(array1, array2)) if a != b]
    return differences