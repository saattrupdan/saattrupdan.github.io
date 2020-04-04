---
layout: post
mathjax: true
title: Quantile regression forests
meta-description: 
---

A random forest is an incredibly useful and versatile tool in a data scientists' toolkit, and is one of the more popular non-deep models that are being used in industry today. 
If we now want our random forests to also output its uncertainty, it would seem that we are forced to go down the [bootstrapping](https://saattrupdan.github.io/2020-03-01-bootstrap-prediction/) route,
as the [quantile approach](https://saattrupdan.github.io/2020-03-09-quantile-regression/) we saw last time relied on the model learning through gradient descent, which random forests aren't.

The bootstrapping would definitely work, but we would be paying a price at inference time. 
Say I have a random forest consisting of 1,000 trees and I'd like to make 1,000 bootstrapped predictions to form a reasonable prediction interval.
Naively, to be able to do that we'd have to make a million decision tree predictions _for every_ prediction we'd like from our model, which can cause a delay that the users of the model wouldn't be too happy about.

In this post I'll describe a surprisingly simple way of tweaking a random forest to enable to it make quantile predictions, which eliminates the need for bootstrapping. This is all from Meinshausen's 2006 paper ["Quantile Regression Forests"](http://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf).

This post is part of my series on quantifying uncertainty:
  1. [Confidence intervals](https://saattrupdan.github.io/2020-02-20-confidence/)
  2. [Parametric prediction intervals](https://saattrupdan.github.io/2020-02-26-parametric-prediction/)
  3. [Bootstrap prediction intervals](https://saattrupdan.github.io/2020-03-01-bootstrap-prediction/)
  4. [Quantile regression](https://saattrupdan.github.io/2020-03-09-quantile-regression/)
  5. Quantile regression forests


## Regression trees with a twist

Let's take a step back and remind ourselves how vanilla random forests work. 
Random forests are simply a collection of so-called decision trees, where we train each decision tree on a bootstrapped resample of the training data set. 
A decision tree is basically just a flow chat diagram. Here's an example of one:

![alt](src)

I won't go into the construction algorithm of decision trees here, as that algorithm is exactly the same in the quantile case; see e.g. Section 9.2 in [Elements of Statistical Learning](https://web.stanford.edu/~hastie/Papers/ESLII.pdf). The rough idea is that we choose the feature and threshold that best separates the target values of the data.
From such a tree, we can now easily figure out which leaf a given input belongs to, by simply answering the yes/no questions all the way down. Super simple.

A _quantile_ decision tree is different when we focus on what happens after having found the correct leaf.
During training, there might have been several training instances ending up in a given leaf, each with their own associated target value.
So what value do we assign to this new instance? In a normal decision tree we simply take the _mean_ of those target values from the training set, so that every leaf has a single target value associated with it.

![alt](src)

The crucial question that Meinshausen asks is whether we can use _all_ of the information in leaves to estimate the _distribution_ of the target values, rather than simply getting a point value.
And the answer is yes! Namely, given a new input variable $x$, we traverse the tree to find the leaf node it belongs to, and then simply look at the distribution of target values present in that leaf.

![alt](src)

That's our estimate for the predictive distribution, which Meinshausen shows is asymptotically a consistent estimate, given a few regularity conditions.
These regularity conditions are quite reasonable, and for instance require that the leaves are approximately evenly populated, that every feature has a chance of affecting the layout of the tree, and that the true distribution is sufficiently continuous.

A **Quantile Regression Forest** is then simply an ensemble of quantile decision trees, each one trained on a bootstrapped resample of the data set, exactly like with random forests.


## Comparison experiments


