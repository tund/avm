Approximation Vector Machines
===========================

This Python code implement Approximation Vector Machines (AVM), presented in the paper "Approximation Vector Machines for Large-scale Online Learning" accepted at Journal of Machine Learning Research 2017.

The code is tested on Windows-based operating system with Python 2.7. Please make sure that you have installed *python-numpy* and *sklearn* to run the example.

Run the demo using this command
-------------------------------------
	python run_avm.py

Citation
--------

```
@Article{le_etal_jmlr17_avm,
  author   = {Trung Le and Tu Dinh Nguyen and Vu Nguyen and Dinh Phung},
  title    = {Approximation Vector Machines for Large-scale Online Learning},
  journal  = {Journal of Machine Learning Research (JMLR)},
  year     = {2017},
  volume   = {18},
  number   = {1},
  pages    = {3962--4016},
  abstract = {One of the most challenging problems in kernel online learning is to bound the model size and to promote the model sparsity. Sparse models not only improve computation and memory usage, but also enhance the generalization capacity, a principle that concurs with the law of parsimony. However, inappropriate sparsity modeling may also significantly degrade the performance. In this paper, we propose Approximation Vector Machine (AVM), a model that can simultaneously encourage the sparsity and safeguard its risk in compromising the performance. When an incoming instance arrives, we approximate this instance by one of its neighbors whose distance to it is less than a predefined threshold. Our key intuition is that since the newly seen instance is expressed by its nearby neighbor the optimal performance can be analytically formulated and maintained. We develop theoretical foundations to support this intuition and further establish an analysis to characterize the gap between the approximation and optimal solutions. This gap crucially depends on the frequency of approximation and the predefined threshold. We perform the convergence analysis for a wide spectrum of loss functions including Hinge, smooth Hinge, and Logistic for classification task, and l1, l2, and ϵ-insensitive for regression task. We conducted extensive experiments for classification task in batch and online modes, and regression task in online mode over several benchmark datasets. The results show that our proposed AVM achieved a comparable predictive performance with current state-of-the-art methods while simultaneously achieving significant computational speed-up due to the ability of the proposed AVM in maintaining the model size.},
  keywords = {kernel, online learning, large-scale machine learning, sparsity, big data, core set, stochastic gradient descent, convergence analysis},
  url      = {https://arxiv.org/abs/1604.06518},
  code      = {https://github.com/tund/avm},
}
```
