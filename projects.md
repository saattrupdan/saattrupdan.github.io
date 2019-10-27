---
layout: page
title: Projects
---

## [AutoPoet](https://github.com/saattrupdan/autopoet)
#### Build poems from text sources

Automatically build poems from a given text source. At the moment it build Haiku poems from tweets, where a machine learning model counts the number of syllables in English words.


## [NaturalSelection](https://github.com/saattrupdan/naturalselection)
#### An all-purpose pythonic genetic algorithm

A python package which uses genetic algorithms to maximise a given function. A prime example of this is to hyperparameter optimise neural networks. See [the readme](https://github.com/saattrupdan/naturalselection/blob/master/README.md) for several examples and a more in-depth explanation of the algorithm, or check out the [accompanying blog post](https://saattrupdan.github.io/2019-09-07-naturalselection-easily-evolve-neural-networks/).

![Image showing an example of the evolution of a population](/img/fashion_mnist.png)


## [Scholarly](https://github.com/saattrupdan/scholarly)
#### Classification of scientific papers

This project aims to classify the category of scientific papers, based on the subject classification on [arxiv.org](https://arxiv.org).

From the ArXiv API I downloaded a million titles and abstracts, which I cleaned up and lemmatised, and built a deep learning model which takes as input the tf-idf extraction of the texts. The model was found using my `naturalselection` package above on a remote supercomputer. So far I have aggregated the ~130 ArXiv categories to the seven major ones, and the model is predicting these categories with a ~86% validation F1-score.

I am currently working on expanding this to the other categories, as well as building a web app that allows interaction with the model.


## [Athelstan](https://github.com/saattrupdan/athelstan)
#### A comparative analysis of constituencies in England and Wales

This project was a first attempt at a cluster analysis of constituencies in England and Wales based on eight different variables related to demographics, business, qualifications and amenities. A use case would be to find similar constituencies as a given location in one of these two countries. Data on constituencies and their gps coordinates are from [doogal.co.uk](https://www.doogal.co.uk), and all the other data are from from [Office for National Statistics](https://www.ons.gov.uk).

This project was the capstone project in the IBM applied data science specialisation on Coursera.
