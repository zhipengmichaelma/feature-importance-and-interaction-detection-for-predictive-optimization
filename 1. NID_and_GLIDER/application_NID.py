"""
The function for using NID and GradientNID methods to explain feature interactions in tabular datasets.
"""
from general_utils import *
from neural_interaction_detection import *
from sampling_tubular import *


def application_NID(
    inputs,
    targets, 
    std_scale=False, 
    detector="GradientNID",
    arch=[256, 128, 64], 
    batch_size=100,
    device=torch.device("cpu"),
    weight_samples=False, 
    add_linear=False, 
    l1_const=None, 
    grad_gpu=-1,
    seed=42,
    number=20,
    feature_names=None,
    decimal=4
    ):
    
    """
    Args:
    Function proprocess_data:
    The function for splitting the data into training, validation and test data and reshape them for the following feature interaction detection
        inputs: numpy 2d array. It can be the whole dataset or part of them
        targets: the target data corresponding to inputs
        std_scale: standardizing the processed data by sklearn.preprocessing.StandardScaler()
    Function detect_interactions:
    The function for detecting the feature interactions by "NID" or "GradientNID"
        detector: "NID" or "GradientNID"
        arch: architecture of the neural networks, default is[256, 128,64]
        batch_size: the batch size of the neural network, default is 100
        device: the object of the device to which torch.Tensor is allocated, default is torch.device("cpu")
        weight_samples: weight sampling with gaussian kernel, default is False
        add_linear: adding the full connection layer, default is False
        l1_const: the const parameter for l1 regularization, default is None
        grad_gpu: if it is -1, the device is cpu, else the device is gpu. Default is -1
        seed: the seed for random state, default is 42
    Function interactions_output:
        number: the number of output interactions, default is 20
        feature_names: the names of the features
        decimal: the number of decimal places of the result, default is 4

    Returns:
        interactions: the feature interaction information
        mlp_loss: the loss of NN
    
    """
    
    # splitting the data into training, validation and test data and reshape them for the following feature interaction detection
    Xs, Ys = proprocess_data(inputs, targets, std_scale=std_scale)
    # feature interaction detection, detector can be "GradientNID" or "NID"
    interactions, mlp_loss = detect_interactions(Xs, Ys, detector=detector, arch=arch, batch_size=batch_size, device=device, weight_samples=weight_samples, 
                                                 add_linear=add_linear, l1_const=l1_const, grad_gpu=grad_gpu, seed=seed)
    if number == "all":                               
        interactions_output(interactions, feature_names=feature_names, decimal=decimal)
    else:
        inter = interactions[:number]
        interactions_output(inter, feature_names=feature_names, decimal=decimal)

    return interactions[:number], mlp_loss
