---
layout: post
mathjax: true
title: NaturalEvolution - A new python package bringing genetic algorithms to the masses
---

In a deep learning project I am currently working on, I faced the inevitable problem of having to tune my hyperparameters. After trying a few dozen combinations, it felt way more like guesswork than anything, and I tried to see what other approaches there were to systematise this process. The two main contenders seem to be [grid search](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search) and [random search](https://en.wikipedia.org/wiki/Random_search), the former searching through a grid of hyperparameters and the latter searching through random combinations of them. My network takes hours to train on my puny gpu-less bog standard laptop, so grid search is definitely ruled out.

After searching around I stumbled across [this excellent blog post by Matt Harvey](https://blog.coast.ai/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164), which is about "evolving" a collection of networks in an intelligent way, inspired by natural evolution. It's essentially a "guided random search", which *roughly* works as follows:

1. Start with a randomly selected population of hyperparameters
2. Train neural networks corresponding to said hyperparameters
3. Set aside a small portion of the better performing networks called *elites*
4. Select a large portion of *breeders* among the entire population, where a high score means a higher chance of being selected --- networks can be chosen twice
5. *Breed* randomly among the breeders by randomly combining hyperparameters of the "parent networks", until the children and elites form a population the same size as the one we started with
6. *Mutate* a small portion of the children by changing a few hyperparameters to something random
7. Go to step 2

In Matt's blog post he supplied code that tuned the amount of layers in the network, the number of neurons in each layer (with each layer having the same number of neurons), the activation function and choice of optimizer.

I really liked the pythonic way he implemented the algorithm, so I decided to try to implement it from scratch myself, adding on several new features along the way. This resulted in my first python package! I call it `naturalevolution`: [here's a link to the github repo](https://github.com/saattrupdan/naturalselection). Here are a few notable features:

* By default it tunes 15 hyperparameters, but this is highly flexible
* It never trains the same model twice
* The training process is parallelised using the `multiprocessing` module
* The breeders and elites are chosen following a distribution such that the higher scoring a network is, the higher chance it has of being selected
* It's highly customisable and can look for maxima for any given function --- hyperparameter tuning for neural networks is just a special case
* It prints some pretty plots :)

To enable a comparison with Matt's algorithm and mine, I'll here go through an example in which we optimise the same data set, [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), which can be fetched and preprocessed as follows:

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
...   ''' Get normalised and scaled MNIST train- and val sets. '''
...   from tensorflow.keras.utils import to_categorical
...   from tensorflow.keras.datasets import cifar10
...   (X_train, Y_train), (X_val, Y_val) = cifar10.load_data()
...   X_train = preprocessing(X_train)
...   Y_train = to_categorical(Y_train)
...   X_val = preprocessing(X_val)
...   Y_val = to_categorical(Y_val)
...   return (X_train, Y_train, X_val, Y_val)

We next define our population consisting of 50 networks and evolve it for 10 generations. We want to be efficient, so we're not training the networks fully. Here I've limited training to three epochs, and if the training is *really* slow then I'm capping it off after 300 seconds as well:

```python
>>> import naturalselection as ns
>>>
>>> fnns = ns.FNNs(
...   size = 50,
...   train_val_sets = cifar10_train_val_sets(),
...   loss_fn = 'binary_crossentropy',
...   score = 'accuracy',
...   output_activation = 'softmax',
...   max_training_time = 300,
...   max_epochs = 3
...   )
>>> 
>>> history = fnns.evolve(generations = 10)
Evolving population: 100%|█████████████████| 10/10 [6:24:28<00:00, 2096.35s/it]
Computing fitness: 100%|███████████████████████| 33/33 [31:46<00:00, 57.98s/it]
```

That took 6.5 hours on my six year old [laptop](https://www.lenovo.com/gb/en/laptops/thinkpad/s-series/s440/). This can probably be optimised by tweaking the number of epochs, training time and size of the population (reducing them to the smallest value still giving us good results), but this will do.

Here are the best performing hyperparameters and a plot of the evolution:

```python
>>> history.fittest
{'genome': {'optimizer': 'adagrad', 'hidden_activation': 'elu',
'batch_size': 16, 'initializer': 'he_normal', 'input_dropout': 0.2,
'neurons0': 0, 'dropout0': 0.2, 'neurons1': 0, 'dropout1': 0.1,
'neurons2': 256, dropout2': 0.1, 'neurons3': 1024, 'dropout3': 0.5,
'neurons4': 0, 'dropout4': 0.0}, 'fitness': 0.4993000030517578}
>>> 
>>> history.plot(
...   title = "Validation accuracy by generation",
...   ylabel = "Validation accuracy"
...   )
```

![alt](https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/naturalselection_data/cifar10_plot.png)

The architecture might seem a bit strange with the zeroes and multiple dropouts, but this corresponds to having neurons [256, 1024] with input dropout 1 - (1-20%)(1-20%)(1-90%) ~ 42% and hidden dropouts [10%, 50%].

Note that we limited training to three epochs, so we can squeeze out some better performance by fully training the fittest network:

```python
>>> # This also saves the model to cifar10_model.h5
>>> best_score fnns.train_best(file_name = 'cifar10_model')
Epoch: 0 - loss: 2.154, acc: 0.341, val_loss: 1.506, val_acc: 0.468: 100%|██████████| 50000/50000 [00:53<00:00, 174.78it/s]
(...)
Epoch: 172 - loss: 0.997, acc: 0.644, val_loss: 1.171, val_acc: 0.584: 100%|██████████| 50000/50000 [00:55<00:00, 152.41it/s]
>>> best_score
0.5836
```

So we end up with a model yielding 58.36% validation accuracy! The evolution took 6.5 hours and the training here took almost three hours, so we're closing in on ten hours in total, but 
