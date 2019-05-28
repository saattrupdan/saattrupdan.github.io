---
layout: post
mathjax: true
title: Forgetful distributions
---

This week we'll deal with memory. More specifically, we'll tackle the question of when a distribution do *not* have any memory whatsoever, meaning that it doesn't depend on past experience in any way. It turns out that there is a *unique* continuous distribution with this property, the *exponential distribution*, and a unique discrete distribution with this property, the *geometric distribution*. Let's dig in.

> **Definition** (Geometric distribution). A random variable $X$ has the **geometric distribution** with parameter $p\in(0,1)$ if it's counting the number of failed [iid](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) [Bernoulli trials](https://en.wikipedia.org/wiki/Bernoulli_trial) with parameter $p$ until it reaches a successful trial. We write $X\sim\text{Geom}(p)$, which has density $f\colon\\\{0,1,2,\dots\\\}\to\mathbb R$ given as $f(k):=(1-p)^kp$.

> **Definition** (Exponential distribution). A random variable $X$ has the **exponential distribution** with parameter $\lambda$ if it counts how long the waiting time is until it reaches a successful Bernoulli trial, where trials are continuously performed with $\lambda$ successes every time unit. We write $X\sim\text{Expo}(\lambda)$, which has density $f\colon(0,\infty)\to\mathbb R$ given as $f(x):=\lambda e^{-\lambda x}$.

I'll get to a couple of examples of both of these distributions in a bit. Also, a disclaimer: I'll mostly be focusing on the exponential distribution in this blog post, primarily to avoid redundancy. Here's a few plots of this distribution, with the associated python code.

{: style="text-align:center"}
![Exponentially distributed random variables](/img/expon.png)

```python
from scipy.stats import expon
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

fig, ax = plt.subplots(1, 4, figsize = (16,3))
lambdas = [.3, .6, 1.0, 1.3]
t = np.arange(0, 10, 0.2)

for (i, lamb) in enumerate(lambdas):
    
    # generate uniformly distributed random variables
    rvs = expon.rvs(size = 500, scale = 1/lamb)
    
    # plot the values of the random variables
    sns.distplot(
      rvs,
      bins = 100,
      color = 'limegreen',
      kde = False, # don't include a kernel density estimator
      ax = ax[i],
      norm_hist = True # normalise the values
      )
    ax[i].axes.set_xlim(0, 10)
    ax[i].axes.set_ylim(0, 1)
    ax[i].axes.plot(t, expon.pdf(t, scale = 1/lamb), 'b--') # plot pdf
    ax[i].title.set_text(f"lambda = {lamb}")

title = "Exponentially distributed random variables"
fig.suptitle(title, y = 1.1, fontsize = 18)
plt.show()
```

As I mentioned, we'll be dealing with the concept of a distribution having *memory*, or lack thereof. We start out with the precise definition and then discuss why it captures the right idea.

> **Defintion** (Memoryless distribution). A distribution $\mathcal D$ is **memoryless** if $X\sim\mathcal D$ implies $P(X\geq s+t\mid X\geq s)=P(X\geq t)$ for all $s,t>0$.

To understand why this could justified as being *memoryless*, take the example of $X$ counting the waiting time for the train. In this case the equation is stating that the probability of waiting at least $t$ minutes is independent of what time it is: we "forget" that we might have waited some time already. Note that $X\sim\text{Expo}(\lambda)$, with $\lambda$ the number of arrivals per minute.

On a more discrete note, we could consider buying scratch cards. Even if we have bought ten scratch cards and won nothing, that does *not* increase the odds of winning if we buy another one! Note that this is following a geometric distribution, as we are counting the number of failed trials until the first success.

An equivalent formulation of this property is that $P(X\geq s+t) = P(X\geq s)P(X\geq t)$, which can be seen using [Bayes' Rule](https://en.wikipedia.org/wiki/Bayes%27_theorem): we firstly have that

$$ P(X\geq s+t \mid X\geq s) = \frac{P(X\geq s+t)P(X\geq s \mid X\geq s+t)}{P(X\geq s)} = \frac{P(X \geq s+t)}{P(X\geq s)}, $$

so that if $\mathcal D$ is memoryless then the lefthand side is $P(X\geq t)$, yielding

$$ P(X\geq s+t) = P(X\geq s)P(X\geq t), $$

and if this equation holds then Bayes' rule applied to the above implies that

$$ P(X\geq s+t \mid X\geq s) = P(X\geq t) $$

The two examples mentioned above show that the exponential and geometric distributions are both memoryless. To show that they're the *unique* discrete and continuous distribution with this property we thus need to show that any given memoryless distribution must be one of the two. In showing this we encounter a healthy mix of calculus and differential equations, so buckle up and I'll try my best to go through it step by step.



> **Theorem.** The exponential distribution is the unique memoryless continuous distribution on $(0,\infty)$, and the geometric distribution is the unique memoryless discrete distribution on $\\\{0,1,2,\dots\\\}$.

We'll just focus on the exponential distribution here. Assume that we have some positive random variable $X\sim\mathcal D$ such that $\mathcal D$ is memoryless. We want to show that $\mathcal D = \text{Expo}(\lambda)$ for some $\lambda$.

Let $F$ be the CDF of $\mathcal D$ and define $G\colon(0,\infty)\to\mathbb R$ as $G(x):=P(X>x)$. Since $F=1-G$, we need to show that $G(x)=e^{-\lambda x}$ for some $\lambda$, since the CDF for $\text{Expo}(\lambda)$ is precisely $x\mapsto 1-e^{-\lambda x}$, which can be seen by the following calculation:

$$ \int_0^x f(y)dy = -\int_0^x-\lambda e^{-\lambda y}dy = -(e^{-\lambda x} - e^{-\lambda\cdot 0}) = 1 - e^{-\lambda x}. $$

We established above that $G(s+t)=G(s)G(t)$, so if we differentiate with respect to $s$ (which is possible as $X$ is continuous, making $G$ differentiable), we get that $G'(s+t)=G'(s)G(t)$, so setting $s=0$ and defining $c:=G'(0)$ and $y:=G(t)$, we arrive at

$$ y' = G'(t) = G'(0+t) = G'(0)G(t) = cG(t) = cy.  $$

This is a [separable differential equation](https://www.khanacademy.org/math/ap-calculus-ab/ab-differential-equations-new/ab-7-6/a/applying-procedures-for-separable-differential-equations) with $\tfrac{dy}{dt} = cy$, so we do the separation and integrate:

$$ \log(y) = \int \frac{1}{y}dy = \int cdt = ct + C $$

for some constant $C$, and setting $K:=e^C$ this means that $y = e^{ct+K} = Ke^{ct}$. As $G(0) = P(X>0) = 1$ we get that $K = \tfrac{G(0)}{e^{c\cdot 0}} = 1$.

This means that $y = e^{ct}$, so if we choose $\lambda := -c$ we get what we want: $G(t) = e^{-\lambda t}$. Note that this makes sense, i.e. that $\lambda > 0$, because $G$ is decreasing, so that $c = G'(0) < 0$. **QED**



So whenever we have data which seems to be memoryless, then there's a *unique* choice for the distribution: exponential if we're looking for a continuous one, and geometric if we want to be discrete. Hoorah!
