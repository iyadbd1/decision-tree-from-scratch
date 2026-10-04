from collections import Counter
import random

import numpy as np


def majority_vote(classes):
    ctr = Counter(classes)
    highest_count = max(ctr.values())
    modes = [item for item, count in ctr.items() if count == highest_count]
    return random.choice(modes)


def predict_proba_classes(classes):
    if len(classes) == 0:
        return np.array([], dtype=float), np.array([], dtype=int)

    counter = Counter(classes)
    ordered_classes = sorted(counter.keys())
    probs = np.array([counter[cl] / len(classes) for cl in ordered_classes], dtype=float)
    return probs, ordered_classes
