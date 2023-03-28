"""
The class for using NID and GradientNID methods to explain feature interactions in tabular datasets.
Firstly, for each data instance, follow the sampling step in lime model. Then train the new local dataset to get the local feature interaction information.
Lastly, combine the local interpretations and rank them to get the global explanations.
"""
import torch.multiprocessing as multiprocessing
from collections import Counter

from general_utils import *
from neural_interaction_detection import *
from sampling_tubular import *


class application_GLIDER(object):
    def __init__(
        self, 
        dataset, 
        feature_names=None, 
        class_names=None, 
        random_state=42, 
        model=None, 
        num_samples=5000,
        sampling_method="Gaussian",
        detector="GradientNID",
        arch=[256, 128, 64], 
        batch_size=100,
        device=torch.device("cpu"),
        weight_samples=False, 
        add_linear=False, 
        l1_const=None, 
        grad_gpu=-1,
        seed=42
        ):
        """Init function

        Args:
            dataset: the input dataset
            feature_names: the names of features
            class_names: the names of targets
            random_state: the random seed for sampling
            model: the model used to train the dataset
            num_samples: size of the neighborhood to learn the linear model in lime algorithm
            sampling_method: 'gaussian' or 'lhs', default is 'gaussian'
            detector: "NID" or "GradientNID"
            arch: architecture of the neural networks, default is[256, 128,64]
            batch_size: the batch size of the neural network, default is 100
            device: the object of the device to which torch.Tensor is allocated, default is torch.device("cpu")
            weight_samples: weight sampling with gaussian kernel, default is False
            add_linear: adding the full connection layer, default is False
            l1_const: the const parameter for l1 regularization, default is None
            grad_gpu: if it is -1, the device is cpu, else the device is gpu. Default is -1
            seed: the seed for random state, default is 42
        """

        self.data = dataset
        self.feature_names = feature_names
        self.class_names = class_names
        self.random_state = random_state
        self.model = model
        self.num_samples = num_samples
        self.sampling_method = sampling_method
        self.detector = detector
        self.arch = arch
        self.batch_size = batch_size
        self.device = device
        self.weight_samples = weight_samples
        self.add_linear = add_linear
        self.l1_const = l1_const
        self.grad_gpu = grad_gpu
        self.seed = seed
     
        

    # function prepared for multiprocessing
    def inter_detection(self, data_instance):   
        """
        local explanations for feature interactions
        Args:
            data_instance: the instance to be perturbed and explained
        Returns:
            interactions: the local feature interaction information
        """
        num_samples = self.num_samples
        detector = self.detector
        explainer = sampling_tubular(self.data, feature_names=self.feature_names, class_names=self.class_names, random_state=self.random_state)
        Xs, Ys = explainer.data_inverse(data_instance, model=self.model, num_samples=num_samples, sampling_method=self.sampling_method)
        interactions, mlp_loss = detect_interactions(Xs, Ys, detector=detector, arch=self.arch, batch_size=self.batch_size, device=self.device, 
                                                     weight_samples=self.weight_samples, add_linear=self.add_linear, l1_const=self.l1_const, 
                                                     grad_gpu=self.grad_gpu, seed=self.seed)
        return interactions

    def multiprocess(self, kernel=20, n=20, printing=True):
        """
        parallel  processing
        Args:
            kernel: the number of processors
            n: the number of output interactions, default is 20
        Returns:
            outputs: the global feature interaction information
        """
        
        dataset = self.data
        pool = multiprocessing.Pool(kernel)
        results = pool.map(self.inter_detection,[data_instance for data_instance in dataset])
        pool.close()
        pool.join()
        b=[]
        for i in range(dataset.shape[0]):
            for j in range(len(results[i])):
                b.append(results[i][j][0])

        res = Counter(b)
        res = dict(res)
        sorted_res = sorted(res.items(), key=operator.itemgetter(1), reverse=True)
        outputs=[]
        for i,count in enumerate(dict(sorted_res).items()):
            inter, num = count
            feature_inter = list()
            for j in inter:
                feature_inter.append(self.feature_names[j])
            if i<n:
                op = "{}: {} count: {}".format(i+1, feature_inter, num)
                if printing is True:
                    print(op)
                outputs.append(feature_inter)

        return outputs







