# RadialBNN

[![Build Status](https://travis-ci.org/danielkelshaw/RadialBNN.svg?branch=master)](https://travis-ci.org/danielkelshaw/RadialBNN)

PyTorch implementation of **[Radial Bayesian Neural Networks](https://arxiv.org/abs/1907.00865)**

This repository provides an implementation of the theory described in
the [Radial Bayesian Neural Networks](https://arxiv.org/abs/1907.00865)
paper. The code provides a PyTorch interface which ensures that the
modules developed can be used in conjunction with other components of
a neural network.

- [x] Python 3.6+
- [x] MIT License

## **Overview:**

The [Radial Bayesian Neural Networks](https://arxiv.org/abs/1907.00865)
paper proposes an alternate variational posterior which scales well to
larger models, unlike that seen in prior posteriors. One of the main 
benefits that the paper brings is the ability to avoid the sampling 
problem seen in mean-field variational inference (MFVI) caused by the 
*soap bubble pathology* of multivariate Gaussians.

Through use of the Radial posterior, samples taken are normalised such 
that they are taken from a direction uniformly selected from a unit
hypersphere. A parameter, `r ~ N(0, 1)`, is used to determine the radius
at which to sample. This avoids the issue of high probability density at
a distance from the mean.  The weight sampling methodology proposed is 
similar to that seen in the [Weight Uncertainty in Neural Netwworks](https://arxiv.org/abs/1505.05424)
paper and is almost as cheap. For more details on this, refer to section
3.1 of the [paper](https://arxiv.org/abs/1907.00865).

## **Example:**

An example of a `Radial BNN` has been implemented in 
`mnist_radial_bnn.py` - this can be run with:

```bash
python3 mnist_radial_bnn.py
``` 

## **References:**

```
@misc{farquhar2019radial,
    title={Radial Bayesian Neural Networks: Beyond Discrete Support In Large-Scale Bayesian Deep Learning},
    author={Sebastian Farquhar and Michael Osborne and Yarin Gal},
    year={2019},
    eprint={1907.00865},
    archivePrefix={arXiv},
    primaryClass={stat.ML}
}
```

##### **[Code](https://github.com/SebFar/radial_bnn)** by Sebastian Farquhar, author of the paper.
###### PyTorch implementation of [Radial Bayesian Neural Networks](https://arxiv.org/abs/1907.00865)<br>Made by Daniel Kelshaw