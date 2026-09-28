from collections import Counter
import random

def majority_vote(classes):
    ctr = Counter(classes)
    highest_count = ctr.most_common()[0][1]
    modes = [item for item, count in ctr.items() if count == highest_count]
    prediction = random.choice(modes)
    return prediction

def predict_proba_classes(classes):
    counter = Counter(classes)
    n = len(counter)
    probs = [counter.get(cl) / n for cl in counter.keys()]
    return probs