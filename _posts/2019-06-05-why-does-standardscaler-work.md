---
layout: post
mathjax: true
title: Why does StandardScaler work?
---

The normal distribution. Gaussian distribution. Bell curve. [The ideal has many names](https://www.goodreads.com/quotes/7745235-the-ideal-has-many-names-and-beauty-is-but-one). But what *is* so special about this distribution? Answering this question turns out to also give justification for Scikit-Learn's `StandardScaler`! Let's get crackin'.

Let's start by introducing our main character. A random variable $X$ has the **standard normal distribution** if its density function $\varphi\colon\mathbb R\to\mathbb R$ is given as

$$ \varphi(z) = \frac{1}{\sqrt{2\pi}}e^{-z^2/2}, $$

where the $\tfrac{1}{\sqrt{2\pi}}$ is there solely to ensure that it integrates to 1. We write $X\sim\mathcal N(0,1)$, because one can show that $EX = 0$ and $\text{Var}(X)=1$ in this case. To generate new normally distributed random variables $Y$ with mean $\mu$ and variance $\sigma^2$ we can simply take $X\sim\mathcal N(0,1)$ and set

$$ Y := \mu + \sigma X, $$

as [can be seen here](https://newonlinecourses.science.psu.edu/stat414/node/166/). Here are a few examples of normally distributed variables, with the associated python code.

{: style="text-align:center"}
![Normal distribution](/img/normal.png)

```python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

mus = [-5, 0, 5]
sigmas = [1, 2, 3]
xs = np.linspace(-10, 10)

fig, ax = plt.subplots(3, 3, sharex = 'col', sharey = 'row', figsize = (10, 10))

for (i, j) in np.ndindex(3, 3):
    Xs = norm.rvs(size = 500, loc = mus[i], scale = sigmas[j])
    sns.distplot(
      Xs,
      bins = 100,
      color = 'limegreen',
      kde = False, # don't include a kernel density estimator
      ax = ax[(i, j)],
      norm_hist = True # normalise the values
    )
    ax[(i, j)].axes.plot(xs, norm.pdf(xs, loc = mus[i], scale = sigmas[j]), 'b--') # plot pdf
    ax[(i, j)].axes.set_xlim(-10, 10)
    ax[(i, j)].axes.set_ylim(0, 0.5)
    ax[(i, j)].axes.set_title(f"mu = {mus[i]} & sigma = {sigmas[j]}", loc='right')
```

Now, moving on to how this is related to a data scientist's work, if we have given an infinite sequence $X_1, X_2, \dots$ of [iid](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) random variables then we define the **n'th sample mean**

$$ \overline X_n := \frac{X_1 + X_2 + \dots + X_n}{n}, $$

which is also just the usual mean that we would be calculating on a given data set (=sample). The difference is of course that here we're talking about *random* variables, so in a data set we'd be working with a particular *instance* of these variables. Working in the more general set up however, we can prove some interesting facts about the sample means.

Firstly, corresponding to our intuition, the sample means have the true mean as expected value:

$$ E\overline X_n = \frac{E(X_1+\dots +X_n)}{n} = \frac{EX_1+\dots +EX_n}{n} = \mu. $$

Note that here we used that all our random variables have the same mean, as we're assuming that they're all following the same distribution. We also used that [expectation is linear](https://brilliant.org/wiki/linearity-of-expectation/).

A stronger result is true in fact, as the following famous theorem shows that the sample means *converge* to the true mean:

> **Theorem** ([Law of Large Numbers](https://terrytao.wordpress.com/2008/06/18/the-strong-law-of-large-numbers/)). Let $X_1, X_2, \dots$ be iid random variables with mean $\mu$. Then $\overline X_n \to \mu$ [almost surely](https://en.wikipedia.org/wiki/Convergence_of_random_variables#Almost_sure_convergence).

Moving from the mean to the variance, we can again easily compute the variance of the sample means:

$$ \text{Var}(\overline X_n) = \frac{\text{Var}(X_1+\dots +X_n)}{n^2} = \frac{\text{Var}(X_1)+\dots +\text{Var}(X_n)}{n^2} = \frac{\sigma^2}{n}, $$

where $\sigma^2$ is the true variance of the $X_n$'s. So the standard deviation of the $n$'th sample mean is therefore $\tfrac{\sigma}{\sqrt{n}}$.

Now, recall that we say that we **standardise** a random variable when we subtract the mean and divide by the standard deviation: this is precisely what Scikit-Learn's `StandardScaler` is doing. This fundamental theorem is both explaining why we care about the normal distribution, and why `StandardScaler` is doing a sensible thing.

> **Theorem** ([Central Limit Theorem](https://math.tutorvista.com/statistics/central-limit-theorem.html)). Let $X_1, X_2, \dots$ be iid random variables with mean $\mu$ and variance $\sigma^2$. Then
>
> $$ \sqrt{n}\left(\frac{\overline X_n-\mu}{\sigma}\right)\to\mathcal N(0,1)\text{ in distribution}. $$

But note that the this object, $\sqrt{n}\sfrac{\overline X_n-\mu}{\sigma}$, is *precisely* the same thing as subtracting the mean ($\mu = E\overline X_n$) and dividing by the standard deviation ($\sigma = \sqrt{\text{Var}(\overline X_n)}=\sqrt{\sfrac{\sigma^2}{n}}=\sfrac{\sigma}{\sqrt{n}}$). In other words, it's *precisely* the result of applying `StandardScaler` to our data! So, said more simply:

{: style="text-align:center"}
*When we apply `StandardScaler` to our data then we approximate a standard normal distribution*

The more samples we have, the closer the distribution will look like a standard bell curve, which will make training machine learning models more efficient. Hoorah!
