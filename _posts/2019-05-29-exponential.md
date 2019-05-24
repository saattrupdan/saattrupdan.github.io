---
layout: post
mathjax: true
title: Forgetful distributions
---

-intro-

> **Definition** (Geometric distribution). A random variable $X$ has the **geometric distribution** with parameter $p\in(0,1)$ if it's counting the number of failed iid Bernoulli trials with parameter $p$ until it reaches a successful trial. We write $X\sim\text{Geom}(p)$, which has density $f\colon\\\{0,1,2,\dots\\\}\to\mathbb R$ given as $f(k):=(1-p)^kp.

> **Definition** (Exponential distribution). A random variable $X$ has the **exponential distribution** with parameter $\lambda$ if it counts how long the waiting time is until it reaches a successful Bernoulli trial, where trials are continuously performed with $\lambda$ successes every time unit. We write $X\sim\text{Expo}(\lambda)$, which has density $f\colon(0,\infty)\to\mathbb R$ given as $f(x):=\lambda e^{-\lambda x}$.

-comment-

{: style="text-align:center"}
![Exponentially distributed random variables](/img/expon.png)

```python
from scipy.stats import expon
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

fig, ax = plt.subplots(1,4, figsize=(16,3))
lambdas = [.3, .6, 1.0, 1.3]
t = np.arange(0, 10, 0.2)

for (i, lamb) in enumerate(lambdas):
    
    # generate uniformly distributed random variables
    rvs = expon.rvs(size = 500, scale = 1/lamb)
    
    # plot the values of the random variables
    sns.distplot(rvs, bins=100, color='limegreen', kde=False, ax=ax[i], norm_hist=True)
    ax[i].axes.set_xlim(0, 10)
    ax[i].axes.set_ylim(0, 1)
    ax[i].axes.plot(t, expon.pdf(t, scale = 1/lamb), 'b--')
    ax[i].title.set_text(f"lambda = {lamb}")

fig.suptitle("Exponentially distributed random variables", y=1.1, fontsize=18)
plt.show()
```

> **Defintion** (Memoryless distribution). A distribution $\mathcal D$ is **memoryless** if $X\sim\mathcal D$ implies $P(X\geq s+t\mid X\geq s)=P(X\geq t)$ for all $s,t>0$.

To understand why this could justfied as being *memoryless*, take the example of $X$ counting the waiting time for the train. In this case the equation is stating that the probability of waiting at least $t$ minutes is independent of what time it is: we "forget" that we might have waited some time already. Note that $X\sim\text{Expo}(\lambda)$, with $\lambda$ the number of arrivals per minute.

On a more discrete note, we could consider buying scratch cards. Even if we have bought ten scratch cards and won nothing, that does *not* increase the odds of winning if we buy another one! Note that this is following a geometric distribution, as we are counting the number of failed trials until the first success.

The above two examples show that the exponential and geometric distributions are both memoryless. It turns out that they are the *unique* discrete and continuous distribution with this property.

> **Theorem.** The exponential distribution is the unique memoryless continuous distribution on $(0,\infty)$, and the geometric distribution is the unique memoryless discrete distribution on $\\\{0,1,2,\dots\\\}$.

-proof-
