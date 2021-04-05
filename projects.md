---
layout: page
title: Projects
meta-description: Personal data science projects by Dan Saattrup Nielsen, involving getting uncertainty estimates of machine learning model predictions, classification of scientific papers, counting syllables in English words, implementing genetic algorithms to evolve neural networks, and performing a cluster analysis of constituencies in England and Wales.
---

## [Doubt](https://github.com/saattrupdan/doubt)
#### Python package that allows uncertainty estimates of machine learning models  --- [blog post](https://saattrupdan.github.io/2021-04-04-doubt/)

This Python package implements various algorithms that allows machine learning
practitioners to get uncertainty estimates from their machine learning models.
This includes wrapping their existing models in a bootstrap wrapper, as well as
implementing stand-alone models such as linear quantile regression and quantile
regression trees.


## [Scholarly](https://github.com/saattrupdan/scholarly)
#### Classification of scientific papers --- [blog post](https://saattrupdan.github.io/2020-01-21-scholarly/)

This project aims to classify the category of scientific papers, based on the subject classification on [arxiv.org](https://arxiv.org). Try the live demo here:

<center>
  <a href="https://www.ttilt.dk/scholarly">
    ttilt.dk/scholarly
  </a>
</center>

From the ArXiv API I scraped ~1.3 million titles and abstracts, which was stored in a SQLite database. On this text corpus I trained my own [FastText](https://fasttext.cc/) bigram word vectors. My classification model was an attention-based recurrent neural network, similar to [SHA-RNN](https://arxiv.org/abs/1911.11423), to classify the texts into the 148 subject categories.

The subject categories can be organised in 6 different 'master categories', such as Mathematics and Physics, and I developed my own loss function to take this into account, so that even when the model mislabels a text then it will assign a label within the same master category. The loss function also takes class imbalance into account, both on the category- and master category level.

Every text can be assigned to multiple categories, making this a multilabel classification problem. A metric relevant to such a task is the sample-average F1 score, which computes the F1 score of the predictions associated to every sample, with the final score being the average of these F1 scores. In a multiclass scenario this would coincide with the accuracy, but with multiple labels it gives more information as it rewards the model for partial results.

The best model achieved a ~93% and ~65% validation sample-average F1 score on the master categories and all the categories, respectively. This was trained on the University of Bristol's compute cluster [BlueCrystal Phase 4 compute cluster](https://www.acrc.bris.ac.uk/acrc/phase4.htm), on a single GPU node for a day.


## [AutoPoet](https://github.com/saattrupdan/autopoet)
#### Build poems from text sources --- [blog post](https://saattrupdan.github.io/2019-11-11-syllables/)

Automatically build Haiku poems from a given text source. To ensure that the syllables are counted correctly a recurrent neural network has been trained to syllabise English words with a 97% accuracy. This model was trained on The [Gutenberg Moby Hyphenator II dataset](http://onlinebooks.library.upenn.edu/webbin/gutbook/lookup?num=3204), consisting of ~170k hyphenated English words.


## [NaturalSelection](https://github.com/saattrupdan/naturalselection)
#### An all-purpose Pythonic genetic algorithm --- [blog post](https://saattrupdan.github.io/2019-09-07-naturalselection/)

A Python package which uses genetic algorithms to maximise a given function. A prime example of this is to hyperparameter optimise neural networks. See [the readme](https://github.com/saattrupdan/naturalselection/blob/master/README.md) for several examples and a more in-depth explanation of the algorithm.

![Image showing an example of the evolution of a population](/img/fashion_mnist.png)


## [Athelstan](https://github.com/saattrupdan/athelstan)
#### A comparative analysis of constituencies in England and Wales

This project was a first attempt at a cluster analysis of constituencies in England and Wales based on eight different variables related to demographics, business, qualifications and amenities. A use case would be to find similar constituencies as a given location in one of these two countries. Data on constituencies and their gps coordinates are from [doogal.co.uk](https://www.doogal.co.uk), and all the other data are from from [Office for National Statistics](https://www.ons.gov.uk).

This project was the capstone project in the IBM applied data science specialisation on Coursera.
