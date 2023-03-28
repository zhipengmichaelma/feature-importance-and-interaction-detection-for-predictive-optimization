"""
The function for retraining with new dataset and picking the important features again
"""
import numpy as np 
from itertools import combinations
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score


def retraining(
    table, 
    model, 
    training_data, 
    training_target,
    test_data, 
    test_target, 
    task='regression',
    method='rmse and r2',
    multi_class='macro',
    rmse=None, 
    r2=None, 
    accuracy=None, 
    f1=None,
    n=5):
    """
    Args:
        table: the feture importance table
        model: the model used before
        training_data: the new training data
        training_target: the targets of training data
        test_data: the new test data
        test_target: the targets of the teat data
        task: 'regression', 'binary_classification' or 'multi_classification'
        method: 'rmse and r2', 'rmse' or 'r2' in regression task; 'acc and f1', 'acc' or 'f1' in classification tasks
        multi_class: 'micro', 'macro', 'samples' or 'weighted', default is 'macro'
        rmse: root mean squared error
        r2: r2 score
        accuracy: accuracy score
        f1: f1 score
        n: the number of candidate features to be deleted
    Returns:
        rmse_best: the optimized rmse score
        r2_best: the optimized r2 score
        acc_best: the optimized accuracy score
        f1_best: the optimized f1 score
        feature_deleted: the deleted unimportant features
    """

    rmse_best = rmse
    r2_best = r2
    acc_best = accuracy
    f1_best = f1
    feature_deleted = []
    index = list(table.index)

    for i in range(n):
        c = list(combinations(index[:n], i+1))
        num = len(c)
        for j in range(num):
            features = list(c[j])
            train_new= np.delete(training_data, features, axis=1)
            test_new = np.delete(test_data, features, axis=1)
            model.fit(train_new, training_target)
            prediction = model.predict(test_new)

            if task == 'regression':
                rmse_new = np.sqrt(mean_squared_error(prediction, test_target))
                r2_new = r2_score(prediction, test_target)
                if method == 'rmse and r2' and rmse_new <= rmse_best and r2_new >= r2_best:
                    rmse_best = rmse_new
                    r2_best = r2_new
                    feature_deleted = features
                if method == 'rmse' and rmse_new <= rmse_best:
                    rmse_best = rmse_new
                    feature_deleted = features
                if method == 'r2' and r2_new >= r2_best:
                    r2_best = r2_new
                    feature_deleted = features
            
            if task == 'binary_classification':
                acc_new = accuracy_score(prediction, test_target)
                f1_new = f1_score(prediction, test_target)
            if task == 'multi_classification':
                acc_new = accuracy_score(prediction, test_target, average=multi_class)
                f1_new = f1_score(prediction, test_target, average=multi_class)    
            if method == 'acc and f1' and acc_new>=acc_best and f1_new>=f1_best:
                acc_best = acc_new
                f1_best = f1_new
                feature_deleted = features
            if method == 'acc' and acc_new>=acc_best:
                acc_best = acc_new
                feature_deleted = features
            if method == 'f1' and f1_new>=f1_best:
                f1_best = f1_new
                feature_deleted = features

    feature_deleted = list(table.features[feature_deleted].values)

    if method == 'rmse and r2':
        return rmse_best, r2_best, feature_deleted
    if method == 'rmse':
        return rmse_best, feature_deleted
    if method == 'r2':
        return r2_best, feature_deleted
    if method == 'acc and f1':
        return acc_best, f1_best, feature_deleted
    if method == 'acc':
        return acc_best, feature_deleted
    if method == 'f1':
        return f1_best, feature_deleted










           


