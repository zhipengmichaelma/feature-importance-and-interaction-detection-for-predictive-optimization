"""
The function for creating a table to show feature importance interpretation
"""

import pandas as pd

def feature_importance_table(sbmodular, feature_names=None, threshold=0.1):
    """
    Args:
        sbmodular: submodular pick in lime model
        feature_names: the names of features 
        threshold: the threshold to pick up important features
    Returns:
        sorted_count: the table showing the feature importance in ascending order
    """

    maps = [exp.as_map() for exp in sbmodular]
    n = len(maps)
    map_sp = [0]*n    # here 'n' is the number of picked instances
    for i in range(n):
        map_sp[i] = list(maps[i].values())[0]
        
# table of the ranked important features. features are picked again according to the weights threshold.
    """
    weights: The sum of the absolute number of the feature-importance weights of all data instances from Submodular Pick after the reselection.
    counts: The number of occurences of the important features in the instances
    """
    m = len(list(feature_names))
    data = {'features': feature_names, 'weights': [0.]*m, 'counts': [0]*m}
    count = pd.DataFrame(data)
    for i in range(n):
        for j in range(m):
            k = map_sp[i][j][0]  # the index of j-th feature in i-th instance
            v = map_sp[i][j][1]  # the weight of j-th feature in i-th instance
            count.weights[k] += abs(v)
            if abs(v)> threshold:
                count.counts[k] += 1

    sorted_count = count.sort_values(by=['weights'])
    return sorted_count