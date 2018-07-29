# npde

This repository contains a Python implementation of npde - a nonparametric model for learning unknown differential equations. The paper we describe the model is on [arxiv](https://arxiv.org/abs/1807.05748). This repository also overrides the MATLAB implementation of npODE model for learning ordinary differential equations (the old MATLAB implementation is [here](https://github.com/cagatayyildiz/npode)).

## Python Code
The implementation is in Python3.5, and it requires [TensorFlow(1.6.0+)](https://www.tensorflow.org/) and [GPflow(1.1+)](https://github.com/GPflow/GPflow).

Fitting our model on some data is as simple as follows:
```python
npde = npde_fit(sess, t, Y, model='sde') # parameters are the TensorFlow session, time points and observations. model could be ode/sde
```

It is also possible to predict the future
```python
path = npde.predict(x0,t) # x0 and t are the time points and the initial value
```
and sample from the learned model
```python
samples = npde.sample(Nw,x0,t) # Nw is the number of samples
```

``npde.ipynb`` contains some more details, including three example fits where the underlying dynamics are governed by an ODE, SDE with constant diffusion, and SDE with state dependent diffusion. Below are ODE and constant diffusion fits:

### Example ODE Fit
![ODE](etc/ode.png)

### Example SDE Fit
![SDE](etc/sde.png)

The wrapper function ``npde_fit`` is where the parameters, kernel, and posterior are initialized, and also where optimization is performed - so perhaps a good starting point to read the code. This function takes the following as optional inputs:
 * ``sf``: drift signal variance
 * ``ell``: drift lengthscale
 * ``sfg``: diffusion signal variance
 * ``ellg``: diffusion lengthscale
 * ``W``: inducing point grid width
 * ``Nw``: number of random draws used for optimizing SDEs
