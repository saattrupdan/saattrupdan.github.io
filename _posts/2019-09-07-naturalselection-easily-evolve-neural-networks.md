---
layout: post
mathjax: true
title: NaturalSelection - a new python package to easily evolve neural networks
---

In a deep learning project I am currently working on, I faced the inevitable problem of having to tune my hyperparameters. After trying a few dozen combinations it felt way more like guesswork than anything and I decided to be more systematic, which eventually led to the development of my python package `NaturalEvolution`, which approaches this problem in an intelligent manner with a simple interface. [Here's the github repo](https://github.com/saattrupdan/naturalselection).

Before it got to that stage, I looked around to see what approaches there were to systematise this process. The two main contenders seemed to be [grid search](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search) and [random search](https://en.wikipedia.org/wiki/Random_search), the former searching through a grid of hyperparameters and the latter searching through random combinations of them. My network takes hours to train on my puny GPU-less bog standard [laptop](https://www.lenovo.com/gb/en/laptops/thinkpad/s-series/s440/), so a grid search was quickly ruled out.

After searching around I stumbled across [this excellent blog post](https://blog.coast.ai/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164) by Matt Harvey, which is about "evolving" a collection of networks in an intelligent way, inspired by natural selection. It's essentially a "guided random search", which *roughly* works as follows:

1. Start with a randomly selected *population* of neural networks
2. Train all the networks in the population
3. Set aside a small portion of the better performing networks called *elites*
4. Select a large portion of *breeders* among the entire population
5. *Breed* randomly among the breeders by randomly combining hyperparameters of the "parent networks", until the children and elites form a population the same size as the one we started with
6. *Mutate* a small portion of the children by changing some of their hyperparameters
7. This constitutes a *generation* of the evolution. Go to step 2 to proceed with the next one

In [Matt's blog post](https://blog.coast.ai/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164) he supplied code that tuned the amount of layers in the network, the number of neurons in each layer (with each layer having the same number of neurons), the activation function and choice of optimizer. I really liked the pythonic way he implemented the algorithm, so I decided to try to implement it from scratch myself, adding on several new features along the way.

This was then what resulted in my first python package! I call it `NaturalSelection`. Check out the [readme](https://github.com/saattrupdan/naturalselection/blob/master/README.md) if you're interested in my particular implementation of the algorithm and some more examples. Here are a few notable features:

* By default it tunes 15 hyperparameters (see below), but this is highly flexible
* It never trains the same model twice
* The training process is parallelised using the `multiprocessing` module
* The breeders and elites are chosen following a distribution such that the higher scoring a network is, the higher chance it has of being selected
* It's highly customisable and can look for maxima for any given function --- hyperparameter tuning for neural networks is just a special case
* It prints some pretty plots :)

I wanted to compare my performance with Matt's algorithm to see if I actually improved anything, so let's go through that application together. It's about finding a neural network modelling the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) data set, which classifies images into 10 different categories like "airplane" and "automobile". We start out by fetching the data and doing some standard preprocessing:

```python
>>> def preprocessing(X):
...   ''' Basic normalisation and scaling preprocessing. '''
...   import numpy as np
...   X = X.reshape((-1, np.prod(X.shape[1:])))
...   X = X.astype('float32')
...   X = (X - X.min()) / (X.max() - X.min())
...   X -= X.mean(axis = 0)
...   return X
...   
>>> def cifar10_train_val_sets():
...   ''' Get normalised and scaled CIFAR-10 train- and val sets. '''
...   from tensorflow.keras.utils import to_categorical
...   from tensorflow.keras.datasets import cifar10
...   (X_train, Y_train), (X_val, Y_val) = cifar10.load_data()
...   X_train = preprocessing(X_train)
...   Y_train = to_categorical(Y_train)
...   X_val = preprocessing(X_val)
...   Y_val = to_categorical(Y_val)
...   return (X_train, Y_train, X_val, Y_val)
```

Next is where `NaturalSelection` enters the picture. All we have to do is define an object of the `NNs` class with the parameters we want, representing a population of neural networks. From which we can call its `evolve` method to run the genetic algorithm.

I will be tuning the default set of hyperparameters, which are the following:

* Optimizer, ranging over `adam`, `adamax` and `nadam`
* Initializer, ranging over `lecun_uniform`, `lecun_normal`, `glorot_uniform`, `glorot_normal`, `he_uniform` and `he_normal`
* Activation function in the hidden layers, ranging over `relu` and `elu`
* Batch size, ranging over 16, 32, 64
* Individual neurons in five hidden layers, ranging over 16, 32, 64, 128, 256, 512 and 1024
* Input dropout and individual dropouts after each hidden layer, ranging over 0%, 10%, 20%, 30%, 40% and 50%.

Again, these are merely default values and can be changed by setting `input_dropout`, `hidden_dropout`, `neurons`, `optimizer`, `hidden_activation`, `batch_size`, `initializer` and/or `max_nm_hidden_layers` to range over other values.

Here I set the size of the population to 30 and evolve it for 30 generations, by which I mean that I will be working with 30 neural networks and run the above-mentioned algorithm 30 times. 

All I want to do is train the networks to the point where I can distinguish the good ones from the bad, so I decided to only train them for a single epoch, but to avoid training some networks for *ages* I also set the maximum training time to two minutes. The other parameters are self-explanatory and [completely standard](https://keras.io/examples/cifar10_cnn/):

```python
>>> import naturalselection as ns
>>>
>>> nns = ns.NNs(
...   size = 30,
...   train_val_sets = cifar10_train_val_sets(),
...   loss_fn = 'binary_crossentropy',
...   score = 'accuracy',
...   output_activation = 'softmax',
...   max_training_time = 120,
...   max_epochs = 1
...   )
>>> 
>>> history = nns.evolve(generations = 30)
Evolving population: 100%|██████████████| 30/30 [2:11:11<00:00, 190.63s/it]
Computing fitness: 100%|█████████████████████| 7/7 [04:10<00:00, 56.28s/it]
```

That only took a bit more than two hours on my laptop, which is not too bad and a substantial improvement of the seven hour run of Matt's algorithm, which of course makes sense as I'm only training our networks for a single epoch and even in parallel.

The evolution updates the `nns` population as well as spitting out a `History` object which carries information about the evolution process, just like when we `fit` a Keras model. This allows us to output the genome (i.e. hyperparameter combination) and fitness (= validation accuracy) of the best performing network throughout the evolution, as well as plot the progress:

```python
>>> history.fittest
{'genome': {'optimizer': 'adamax', 'hidden_activation': 'relu',
'batch_size': 64, 'initializer': 'lecun_normal', 'input_dropout': 0.0,
'neurons0': 256, 'dropout0': 0.1, 'neurons1': 0, 'dropout1': 0.0,
'neurons2': 256, 'dropout2': 0.1, 'neurons3': 0, 'dropout3': 0.0,
'neurons4': 256, 'dropout4': 0.1}, 'fitness': 0.4674000144004822}
>>> 
>>> history.plot(
...   title = "Validation accuracy by generation",
...   ylabel = "Validation accuracy"
...   )
```

![Plot of evolution over thirty generations, where the average validation accuracy steadily increases from 30% to 45%, with the maximum rising from 43% to 46% with a handful of small oscillations along the way](https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/naturalselection_data/cifar10_plot.png)

Here the filled area are the accuracies that are one standard deviation away from the mean, which assuming that they're normally distributed would account for ~68% of the population, giving you a rough idea of how homogeneous the population during the evolution.

The architecture might seem a bit strange with all the zeroes, but this corresponds to having neurons [256, 256, 256] with no input dropout and hidden dropouts [10%, 10%, 10%]. Note that since I limited myself to training our models for a single epoch, I can squeeze out some more performance by fully training the fittest network:

```python
>>> # This also saves the model to cifar10_model.h5
>>> best_score = nns.train_best(file_name = 'cifar10_model')
Epoch 0: 100%|██████████████████████| 50000/50000 [00:15<00:00, 958.74it/s]
(...)
Epoch 34: 100%|██████████████████████| 50000/50000 [00:16<00:00, 82.02it/s]
>>> best_score
0.5671
```

So I end up with a model yielding 56.71% validation accuracy, which is slightly better than Matt's score, which makes sense as I'm also tuning more hyperparameters. All in all this ended up taking less than 2.5 hours!

This package is still work in progress, but it's coming close to reaching a stable state (at the time of writing it's at version 0.6). If you think you'll find this useful then I'd appreciate if you could give [the repo](https://github.com/saattrupdan/naturalselection) a star, and feel free to open a ticket if you spot a bug. Pull requests with fixes or new features would also be awesome!

Some things I will be looking into including are at least:
* Built-in support for CNN- and RNN- layers as well, as it currently only works with densely connected "vanilla" layers
* Implementing GPU parallel training
* Searching through network topologies as well, to not limit ourselves to sequential neural networks

You can check out the `dev` branch to see what work is currently in progress.
