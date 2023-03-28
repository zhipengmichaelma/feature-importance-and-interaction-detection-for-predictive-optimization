"""
The function for creating new dataset after getting feature importance and feature interactions information
"""
import numpy as np 

def combination(interactions, training_data, test_data, feature_names, delete=None, num=10):
    """
    Args:
        interactions: the feature interactions interpretation
        training_data: the original training data
        test_data: the original test data
        feature_names: the names of the features
        delete: the unimportant features to be deleted, default is None. If it is not None, for example, the 1st and 2nd features are deleted, delete=[1,2].
        num: the number of selected features
    Returns:
        train_new: the new training data
        test_new: the new test data
        feature_names_new: the new feature names
    """

# creating new features with interactive information by product
    inter_train=[]
    inter_test=[]
    for i in range(num):
        a = []
        for j in range(len(interactions[i][0])):
            a.append(interactions[i][0][j])
            if j==0:
                inter_train.append(training_data[:,a[j]])
                inter_test.append(test_data[:,a[j]])
            inter_train[i] = inter_train[i] * training_data[:,a[j]]
            inter_test[i] = inter_test[i] * test_data[:,a[j]]
    inter_train = np.array(inter_train).T
    inter_test = np.array(inter_test).T

# concating the original dataset and the new interactive dataset
    train_new = training_data.copy()
    test_new = test_data.copy()
    if delete is not None:
        train_new= np.delete(train_new, delete, axis=1)
        test_new= np.delete(test_new, delete, axis=1)
    train_new = np.c_[train_new, inter_train]
    test_new = np.c_[test_new, inter_test]

# extending the feature names
    feature_names_new = feature_names.copy()
    if delete is not None:
        feature_names_new = list(np.delete(feature_names_new, delete))

    feature_names_new += [f"inter_{i+1}" for i in range(num)]
    
    return train_new, test_new, feature_names_new

