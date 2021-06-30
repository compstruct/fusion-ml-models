## Abstract

Many modern convolutional neural neworks (CNNs) rely on bottleneck block structures where the activation tensor is mapped between higher dimensions using an intermediate low dimension, and convolved with depthwise feature filters rather than multi-channel filters. Because most of the computation lies in computing the large dimensional tensors, however, such networks cannot be scaled without significant computation costs.

In this paper, we show how \emph{fusing} the layers inside these blocks can dramatically reduce the multiplication count (by 6--20x) at the cost of extra additions. ReLU nonlinearities are predicted dynamically, and only the activations that survive ReLU contribute to directly compute the output of the block. We also propose FusioNet, a CNN architecture optimized for fusion, as well as ARCHON, a novel accelerator design with a dataflow optimized for fused networks.

When FusioNet is executed on the proposed accelerator, it yields up to 5.8x faster inference compared to compact networks executed on a dense DNN accelerator, and 2.1x faster inference compared to the same networks when pruned and executed on a sparse DNN accelerator.

## Paper Link

[Accelerating DNNs Inference with Predictive Layer Fusion, in ICS'21](https://dl.acm.org/doi/10.1145/3447818.3460378)

## Requirements

Install the followings:

* [TensorFlow-GPU](https://www.tensorflow.org/install/docker) version 2.3.0 (you can pull the TF docker "tensorflow-2.3.0-gpu" for this)
* tensorflow_datasets
* tensorflow_model_optimization
* numpy
* pandas
* [CIFAR10 & CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) datasets



## Overview

We performed our experiments on two datasets, CIFARF10 and 100.

### CIFAR10

We experimented on MobileNetv2, ShuffleNet, MnasNet and our design, FusioNet. Each of the first 3 CNNs has a baseline vanilla & pruned version, as well as a modified
vanilla, pruned & pred version. For these, we first train the vanilla network; then, we perform pruning on the pre-trained vanilla CNN; and finally, if aiming for layer
fusion, we apply  ReLU prediction on the pruned network.

### CIFAR100

We experimented on MobileNetv2, MnasNet and FusioNet. Only FusioNet has a version with ReLU predictor as this is the only network we experimented with layer fusion for
CIFAR100.


## Running Experiments

### Organization

Experiments are organized in terms of dataset, modified vs. baseline and flavor (i.e. pruned, vanilla, pred). Figure below shows the overall structure.

```
cnns
└── cifar10
|    ├── mobilenet
|    |       └── modified
|    |       |      └── vanilla
|    |       |      └── pruned           
|    |       |      └── pred           
|    |       └── baseline
|    |              └── vanilla
|    |              └── pruned           
|    └── mnasnet
|    |       └── modified
|    |       |       └── vanilla
|    |       |       └── pruned           
|    |       |       └── pred           
|    |       └── baseline
|    |               └── vanilla
|    |               └── pruned 
|    └── shufflenet
|    |        └── modified
|    |        |      └── vanilla
|    |        |      └── pruned           
|    |        |      └── pred           
|    |        └── baseline
|    |               └── vanilla
|    |               └── pruned 
|    └── fusionet
|            └── vanilla
|            └── pruned
|            └── pred
|
└── cifar100
    ├── mobilenet
    |            └── vanilla
    |            └── pruned           
    └── mnasnet
    |            └── vanilla
    |            └── pruned 
    └── fusionet
                └── vanilla
                └── pruned 
                └── pred
```

For example, to run the pruning on baseline MobileNet for CIFAR10, go to the directory "cifar10/mobilenet/baseline/pruned" and run the submission script.

### Job Submission

jobs are submitted using submission.sh via sbatch. Number of GPUs, path for python libraries, path to the experiment directory and TF container
can all be set in the submission script. Furthermore, networks can be ran in training or inference mode (using the pre-trained models). To run the training, set the
training parameter in the submission file to True and to run in inference set the training to False and pre-trained to True.


## Details

### Prediction
To implement predictions, we needed to mix TF layers into Keras models. We did it in two ways:
* By creating TF variabls that are initialized with weights and using them with TF conv and operation layers.
* By using the weight constraining feature provided by TF and Keras. Specifically, we create a new constrain 
  class that ternerize weights before actual convolution. This method was used for FusioNet on CIFAR100.

*Notes*

* When using the second method, layers should be named. That's why we use the `pruned_named` model for that method.
* Keras has problem loading models with TF layers in them. In order to observe the actual performance with predictor, you may have to run the predictor training and watch the 
  validation accuracies during the training. 
