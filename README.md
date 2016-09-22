# DenseNet-tensorflow
This repository contains the tensorflow implementation for the paper [Densely Connected Convolutional Networks](http://arxiv.org/abs/1608.06993). 

The code is developed based on Yuxin Wang's implementation of ResNet (https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet).

Citation:

     @article{Huang2016Densely,
     		author = {Huang, Gao and Liu, Zhuang and Weinberger, Kilian Q.},
     		title = {Densely Connected Convolutional Networks},
     		journal = {arXiv preprint arXiv:1608.06993},
     		year = {2016}
     }

## Dependencies:

+ Python 2 or 3
+ TensorFlow >= 0.8
+ [Tensorpack] (https://github.com/ppwwyyxx/tensorpack)

## Train a DenseNet (L=40, k=12) on CIFAR-10+ using

```
python cifar10-densenet.py 
```
In our experiment environment (cudnn v5.1, CUDA 7.5, one TITAN X GPU), the code runs with speed 5iters/s when batch size is set to be 64. The hyperparameters are identical to the original [torch implementation] (https://github.com/liuzhuang13/DenseNet). 


