---
layout: post
mathjax: true
title: The Three Discretes
---

In these first few posts I'll cover a handful of distributions, what interesting properties they have and how they're connected to each other. I'll try to follow standard notation here, so that capital letters $X,Y,Z$ will denote random variables, which are represented by functions $X:\Omega\to\mathbb R$ for some sample probability space $\Omega$. $P(A)$ will be the probability of event $A$ and, as is custom, I will abuse notation and write things like $P(X=0)$ for $P(\{x\mid X(x)=0\})$.

We start out with three discrete distributions that are intimately connected: the *binomial*, *hypergeometric* and *Poisson*. Let's start by getting a feel for what these beasts are measuring. As a warm-up we'll throw in a fourth one: the *Bernoulli distribution*. This is mostly because this distribution is a sort of a "building block" for several other distributions.

## Bernoulli

The easiest way to grasp what the Bernoulli distribution is doing is via the notion of a **Bernoulli trial**. Such a trial is a very simple concept: it's simply an experiment with two outcomes, success and failure. Associated to such a trial is the *success rate*, which is the probability of the experiment resulting in a success. A basic example is the coin flip, where the success rate (of getting heads, say) is 50%. Another example could be whether a volcano will erupt at a given moment, where we'd hope that the success rate is way below 50%.

Now, a random variable $X$ following a Bernoulli distribution with success rate $p$, written $X\sim\text{Bern}(p)$, means that $X$ is the outcome of a single Bernoulli trial. The density of this distribution is $f\colon\{0,1\}\to\mathbb R$, given by $f(k)=(p-1)(k-1)+k$, which is really just a convoluted way of saying that $f(0)=P(X\leq 0)=1-p$ and $f(1)=P(X\leq 1)=1$. Here's a plot:

![Bernoulli distribution](img/bernoulli.png)

```python
from scipy.stats import bernoulli
from matplotlib import pyplot as plt
from random import uniform

p = uniform(0, 1)
X = bernoulli(p)
xs = range(2)

plt.bar(xs, height=X.pmf(xs))
plt.xticks([0,1])
plt.show()
```

## Binomial

A binomial distribution is a simple generalisation of the Bernoulli distribution. Instead of doing a single Bernoulli trial, we do many!

So a random variable $X$ following a binomial distribution with parameters $n$ and $p$, written $X\sim\text{Bin}(n,p)$, simply means that $X$ counts the number of successful trials out of $n$ attempts, each having success rate $p$. The binomial density is then $f\colon\{0,\hdots,n\}\to\mathbb R$, given as

$$ f(k)=\sum\_{m=0}^k {n\choose m}p^m(1-p)^{n-m} $$.

Let's illustrate that in a plot:

**insert image**

## Hypergeometric

asd

## Poisson

asd
