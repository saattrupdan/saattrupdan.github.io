---
layout: post
mathjax: true
title: NaturalSelection - easily evolve neural networks and more
---

In a deep learning project I am currently working on, I faced the inevitable problem of having to tune my hyperparameters. After trying a few dozen combinations, it felt way more like guesswork than anything, and I tried to see what other approaches there were to systematise this process. The two main contenders seem to be [grid search](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search) and [random search](https://en.wikipedia.org/wiki/Random_search), the former searching through a grid of hyperparameters and the latter searching through random combinations of them. My network takes hours to train on my puny GPU-less bog standard [laptop](https://www.lenovo.com/gb/en/laptops/thinkpad/s-series/s440/), so grid search was quickly ruled out.

After searching around I stumbled across [this excellent blog post](https://blog.coast.ai/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164) by Matt Harvey, which is about "evolving" a collection of networks in an intelligent way, inspired by natural selection. It's essentially a "guided random search", which *roughly* works as follows:

1. Start with a randomly selected "population" of neural networks
2. Train all the networks in the population
3. Set aside a small portion of the better performing networks called *elites*
4. Select a large portion of *breeders* among the entire population
5. *Breed* randomly among the breeders by randomly combining hyperparameters of the "parent networks", until the children and elites form a population the same size as the one we started with
6. *Mutate* a small portion of the children by changing some of their hyperparameters
7. Go to step 2

In [Matt's blog post](https://blog.coast.ai/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164) he supplied code that tuned the amount of layers in the network, the number of neurons in each layer (with each layer having the same number of neurons), the activation function and choice of optimizer. I really liked the pythonic way he implemented the algorithm, so I decided to try to implement it from scratch myself, adding on several new features along the way. This resulted in my first python package! I call it `NaturalSelection`: [here's a link to the github repo](https://github.com/saattrupdan/naturalselection), where I also describe the implementation of the algorithm in a bit more detail. Here are a few notable features:

* By default it tunes 15 hyperparameters, but this is highly flexible
* It never trains the same model twice
* The training process is parallelised using the `multiprocessing` module
* The breeders and elites are chosen following a distribution such that the higher scoring a network is, the higher chance it has of being selected
* It's highly customisable and can look for maxima for any given function --- hyperparameter tuning for neural networks is just a special case
* It prints some pretty plots :)

I wanted to compare my performance with Matt's algorithm to see if I actually improved anything, to let's go through that application together. It's about find a neural network modelling the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) data set, which classifies images into 10 different categories like "airplane" and "automobile". We start out by fetching the data and doing some standard preprocessing:

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

Next is where `NaturalSelection` enters the picture. All we have to do is define an object of the `NNs` class with the parameters we want, representing a population of neural networks, from which we can call its `evolve` method to run the genetic algorithm.

Here I set the size of the population to 30 and evolve it for 30 generations. All we want to do is train the networks to the point where we can distinguish the good ones from the bad, and I ended up going for only a single epoch. To avoid some networks that take *ages* to train I also set the maximum training time to two minutes. The other parameters are self-explanatory and completely standard:

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
Evolving population: 100%|██████████████████| 30/30 [2:11:11<00:00, 190.63s/it]
Computing fitness: 100%|█████████████████████████| 7/7 [04:10<00:00, 56.28s/it]
```

That only took a bit more than two hours on my laptop, which is not too bad and a lot faster than Matt's run of his algorithm, which also makes sense as we're only training our networks for one epoch and even in parallel.

The evolution updates the `nns` population as well as spitting out a `History` object which carries information about the evolution process, just like when we `fit` a Keras model. This allows us to output the genome (i.e. hyperparameter combination) and fitness (= validation accuracy) of the best performing network throughout the evolution, as well as plot the progress:

```python
>>> history.fittest
{'genome': {'optimizer': 'adamax', 'hidden_activation': 'relu',
'batch_size': 64, 'initializer': 'lecun_normal', 'input_dropout': 0.0,
'neurons0': 256, 'dropout0': 0.1, 'neurons1': 0, 'dropout1': 0.0,
'neurons2': 256, dropout2': 0.1, 'neurons3': 0, 'dropout3': 0.0,
'neurons4': 256, 'dropout4': 0.1}, 'fitness': 0.4674000144004822}
>>> 
>>> history.plot(
...   title = "Validation accuracy by generation",
...   ylabel = "Validation accuracy"
...   )
```

![Plot of evolution over ten generations, where the average validation accuracy steadily increases from 36% to 42%, with the maximum oscillating between 45% and 48% until it jumps up to 50% on generation 8](https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/naturalselection_data/cifar10_plot.png)

The architecture might seem a bit strange with all the zeroes, but this corresponds to having neurons [256, 256, 256] with no input dropout and hidden dropouts [10%, 10%, 10%]. Note that we limited ourselves to training for a single epoch, so we can squeeze out some better performance by fully training the fittest network:

```python
>>> # This also saves the model to cifar10_model.h5
>>> best_score = nns.train_best(file_name = 'cifar10_model')
Epoch: 0 - loss: 1.6879, acc: 0.3977, val_loss: 1.5072, val_acc: 0.4605: 100%|██████████| 50000/50000 [00:15<00:00, 958.74it/s]
(...)
Epoch: 34 - loss: 0.5293, acc: 0.8099, val_loss: 1.5934, val_acc: 0.5641: 100%|██████████| 50000/50000 [00:16<00:00, 82.02it/s]
>>> best_score
0.5671
```

So we end up with a model yielding 56.71% validation accuracy, which is slightly better than Matt's score, which makes sense as we're tuning more hyperparameters. All in all this took less than 2.5 hours!

This package is still work in progress, but it's coming close to reaching a stable state (at the time of writing it's at version 0.6). If you think you'll find this useful then I'd appreciate if you could give [the repo](https://github.com/saattrupdan/naturalselection) a star, and feel free to open a ticket if you spot a bug. Pull requests with fixes or new features would also be awesome as well!

Some things I think would be great to include are at least:
* Built-in support for CNN- and RNN- layers as well, as it currently only works with densely connected "vanilla" layers
* Implementing GPU parallel training
* Searching through network topologies as well, to not limit ourselves to sequential neural networks

I'll also be working on these things and will be commiting updates on the `dev` branch.
