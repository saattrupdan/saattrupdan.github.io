---
layout: post
mathjax: true
title: Quantile regression forests
meta-description: Quantile regression forests is a way to make a random forest output quantiles and thereby quantify its own uncertainty. This method only requires training the forest once. We compare the QRFs to bootstrap methods on the hourly bike rental data set.
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

![A small flow chart diagram, aka a decision tree, to model how cold it will be tomorrow. The leaves each contain a single value.](/img/qrf-decision-tree.jpg)

I won't go into the construction algorithm of decision trees here, as that algorithm is exactly the same in the quantile case; see e.g. Section 9.2 in [Elements of Statistical Learning](https://web.stanford.edu/~hastie/Papers/ESLII.pdf). The rough idea is that we choose the feature and threshold that best separates the target values of the data.
From such a tree, we can now easily figure out which leaf a given input belongs to, by simply answering the yes/no questions all the way down. Super simple.

A _quantile_ decision tree is different when we focus on what happens after having found the correct leaf.
During training, there might have been several training instances ending up in a given leaf, each with their own associated target value.
So what value do we assign to this new instance? In a normal decision tree we simply take the _mean_ of those target values from the training set, so that every leaf has a single target value associated with it.
However, the values in the leaves giving rise to the above tree might have looked like this:

![Same tree as before, but the leaves now contain many values, each of whose mean corresponds to the previous diagram.](/img/qrf-decision-tree2.jpg)

So by taking the means we potentially lose a lot of information. The crucial question that Meinshausen asks is whether we can use all of the information in the leaves to estimate the _true distribution_ of the target values, rather than simply getting a point value? It turns out that the answer is yes! Namely, given a new input variable $x$, we traverse the tree to find the leaf node it belongs to, and then simply look at the distribution of target values present in that leaf, which will be our estimate for the predictive distribution.

It almost sounds too good to be true that such an estimate would be reasonable. We'll get back the why in the next section, but let's just assume it works for a moment.
Given such an estimate we can now also output quantiles rather than the mean: we simply compute the given quantile out of the target values in the leaf.
A **Quantile Regression Forest (QRF)** is then simply an ensemble of quantile decision trees, each one trained on a bootstrapped resample of the data set, exactly like with random forests.

Note one crucial difference between these QRFs and the quantile regression models [we saw last time](https://saattrupdan.github.io/2020-03-09-quantile-regression/) is that by only training a QRF *once*, we have access to *all* the quantiles at inference time, where in the previous case we would have to train our model separately for every quantile. Also, as we also noted last time, the quantile model is able to deal with heteroscedasticity, which bootstrapping can't really deal with.


## Consistency of the estimate

Meinshausen shows that the CDF estimate described in the previous section *is* asympotically a consistent estimate, given the following conditions, where $n$ is the size of the training set:

  1. The proportion of values in a leaf, relative to all values, is vanishing as $n\to\infty$. Intuitively, this means that the leaves are roughly evenly populated
  2. The minimal number of values in a tree node is growing as $n\to\infty$. This means that our tree depth grows slowly ($O(\log(n)$)
  3. When looking for features at a split, the probability of a feature being chosen is uniformly bounded from below. I.e., we don't "forget" about features
  4. There's a constant $\gamma\in(0, 0.5]$ such that the number of values in a child node is always at least $\gamma$ times the number of values in the parent node. This roughly means that our branches are always "thick"
  5. The true CDF of the predictive distribution is [Lipschitz continuous](https://en.wikipedia.org/wiki/Lipschitz_continuity)

Hopefully you will agree that these are not wild assumptions. 
From these assumptions Meinshausen proved the following, where $F$ is the CDF of the true predictive distribution and $\hat F$ is our estimate as described in the previous section:

> **Theorem (Meinhausen).** Assume the above five conditions hold. Then, for every $x$,
> $$ \sup_{y\in\mathbb R}|\hat F(y\mid X=x) - F(y\mid X=x)| $$
> converges in probability to $0$ as $n\longrightarrow\infty$.

This is thus saying that our CDF estimate $\hat F$ *is* an asymptotically consistent estimate. The curious thing is that this even holds for a single tree, as his proof doesn't use the number of trees for anything.
He mentions this himself, and as random forests with many trees vastly outperform a single tree, he conjectures that a larger forest would increase the convergence rate of the above consistency.
But that's further research to be done!


## Comparison to the bootstrap

The crucial question when it comes to evaluating the QRFs is whether the QRFs have the same performance as the bootstrap methods.
We know that _asymptotically_ both with give perfect prediction intervals, but one might fear that the QRFs take some time to catch up, where by "time" we really mean the size of the data set.
In the previous posts I always used simulated data for these experiments, so to change things up a bit and model some real data.

I chose the [hourly bike-sharing dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset) from UCI, which is a data set with 17,379 rows where the aim is to predict the number of rental bike users from 13 numeric date- and weather features.
The response variable in this data set is also highly heteroscedastic which the quantile method can deal with out of the box, so it gives us a chance to see this in action as well.
Here's the distribution of our response:

![Plot of response. Most values are very close to zero, from where it monotonically decreases, reaching close to zero around ~800 rentals](/img/qrf-response.png)

I trained a QRF and 100 bootstrapped regular random forests on the data set, with 100 trees in each forest.
This gives us the following.

![Plot of sorted prediction intervals of the QRF and the bootstrapped random forest. The variance of the residuals vary between approx 0 and 400, with the quantile intervals capturing this. 
The bootstrap interval is of almost uniform length and thus does not capture the shape of the data, even though it has almost perfect coverage (93.5%) and the quantile intervals have a coverage of 80.9%.](/img/qrf-1leaf.png)

We clearly see that the quantile method (as anticipated) can deal with the heteroscedasticity a lot better than the bootstrap. 
But we also see that the quantile intervals aren't on par with the bootstrapped ones, only having 81% coverage on a 95% prediction interval.
Let's take a closer look at what is happening:

![This plot shows that the samples lying outside the quantile method are primarily when the intervals are close to zero, with the opposite being the case for the bootstrapped intervals.](/img/qrf-coverage-analysis.png)

Aha! This shows that almost all of the values outside the quantile prediction interval being those samples where the interval width is near zero (with the opposite being the case for the bootstrapped versions).
If we compare the distributions of the interval lengths of the samples that are in and out of the prediction intervals, respectively, we get the following.

![We see that the majority of the samples outside the intervals are when the interval widths are very narrow (less than approx 50)](/img/qrf-in-out-interval.png)

We see that it *is* mostly when the intervals are tiny that the samples tend to land outside. We can also see this if we mark the samples that are outside the intervals:

![Same sorted prediction interval plot but without the bootstrap and with the samples marked if they're outside the intervals. Most of those are in the beginning, when the intervals are really narrow](/img/qrf-1leaf-in-out.png)

A simple solution to this, if we *really* care that much about the "hard" coverage (in most cases in practice we wouldn't) then we could simply uniformly pad the quantile interval by 20 units (rental bike users):

![Same plot as the original sorted prediction intervals, but with padding on the quantile intervals. The quantile intervals are still only half as wide as the bootstrap intervals to start out with, and they achieve 94.4% coverage, which is more than the 93.5% coverage of the bootstrap.](/img/qrf-padded.png)

Okay, that's enough of beating that dead horse.
