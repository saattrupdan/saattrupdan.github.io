---
layout: post
mathjax: true
title: Pythonic evolution of neural networks
---

In a deep learning project I am currently working on, I faced the inevitable problem of having to tune my hyperparameters. After trying a few dozen combinations, it felt way more than anything like guesswork, and I tried to see what other approaches there were to systematise this process. The two main contenders seem to be [grid search](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search) and [random search](https://en.wikipedia.org/wiki/Random_search), the former searching through a grid of hyperparameters and the latter searching through random combinations of them. My network takes hours to train on my puny gpu-less bog standard computer, so grid search is definitely ruled out.

After searching around I stumbled across [this excellent blog post by Matt Harvey](https://blog.coast.ai/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164), which is about "evolving" a collection of networks in an intelligent way, inspired by natural evolution. It's essentially a "guided random search", which *very roughly* works as follows:

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

To enable a comparison with Matt's algorithm and mine, I'll here go through an example in which we optimise the exact same hyperparameters on the same data set, in a serialised fashion. After that, I'll demonstrate what we get if we instead go for the default settings, tuning all 15 hyperparameters in a parallel fashion.

The data set we're modelling is [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), which can be fetched and preprocessed as follows:

```python
>>> import naturalselection as ns
>>> from tensorflow.keras.datasets import cifar10
>>> from tensorflow.keras.utils import to_categorical
>>> 
>>> # Standard train and test sets for CIFAR-10
>>> (X_train, Y_train), (X_val, Y_val) = cifar10.load_data()
>>> X_train = ((X_train / 255) - 0.5).reshape((-1, 3072))
>>> Y_train = to_categorical(Y_train)
>>> X_val = ((X_val / 255) - 0.5).reshape((-1, 3072))
>>> Y_val = to_categorical(Y_val)
```

We next define our population consisting of 20 networks and evolve it for 10 generations, identical to Matt's setup. As we're serialising the algorithm in this example I'll set `multiprocessing = False`:

```python
>>> fnns = ns.FNNs(
>>>     size = 20,
>>>     train_val_sets = (X_train, Y_train, X_val, Y_val),
>>>     loss_fn = 'binary_crossentropy',
>>>     score = 'accuracy',
>>>     output_activation = 'softmax',
>>>     max_training_time = 120
>>>     )
>>> 
>>> history = fnns.evolve(generations = 10, multiprocessing = False)
Evolving population: 100%|██████████████████| 10/10 [4:01:44<00:00, 960.32s/it]
Computing fitness for gen 19: 100%|████████████| 16/16 [14:08<00:00, 18.00s/it]
```

Here are the best performing hyperparameters and a plot of the evolution:

```python
>>> history.fittest
{'genome': {'optimizer': 'adam', 'hidden_activation': 'relu',
'batch_size': 1024, 'initializer': 'glorot_normal', 'input_dropout': 0.0,
'neurons0': 128, 'dropout0': 0.0, 'neurons1': 64, 'dropout1': 0.1,
'neurons2': 64, dropout2': 0.2, 'neurons3': 32, 'dropout3': 0.0,
'neurons4': 128, 'dropout4': 0.3}, 'fitness': 0.9748}
>>> 
>>> history.plot(
...   title = "Validation accuracy by generation",
...   ylabel = "Validation accuracy"
...   )
```

![alt](https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/naturalselection_data/cifar10_example.png)

Note that we only trained each network for 60 seconds, so we can squeeze out some better performance by fully training the fittest network:

```python
>>> # This also saves the model to cifar10_model.h5
>>> best_score fnns.train_best(file_name = 'cifar10_model')
Epoch: 0 - loss: 0.277, val_loss: 0.179: 100%|██████████| 60000/60000 [00:31<00:00, 244.79it/s]
(...)
>>> best_score
0.98
```

And that's it!
