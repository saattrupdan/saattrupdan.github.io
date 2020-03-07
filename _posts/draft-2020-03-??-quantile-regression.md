---
layout: post
mathjax: true
title: Quantile Regression
meta-description: 
---

When we are performing regression analysing using complicated predictive models such as neural networks, knowing how *certain* the model is is highly valuable in many cases, especially when the applications are within the health sector.

However, the [bootstrap prediction intervals](https://saattrupdan.github.io/2020-03-01-bootstrap-prediction/) that we covered last time requires us to train the model on a large number of bootstrapped samples, which is unfeasible if training the model takes many hours or days.

Thankfully, there are alternatives. One of those is *quantile regression*, which we'll have a closer look at in this post.

This post is part of my series on quantifying uncertainty:
  1. [Confidence intervals](https://saattrupdan.github.io/2020-02-20-confidence/)
  2. [Parametric prediction intervals](https://saattrupdan.github.io/2020-02-26-parametric-prediction/)
  3. [Bootstrap prediction intervals](https://saattrupdan.github.io/2020-03-01-bootstrap-prediction/)
  4. Quantile regression


## Going beyond the mean

When we are fitting predictive models for regression we tend to use the **mean squared error (MSE)**, or **l2**, loss function

$$ \textsf{MSE}(y, \hat y) := \frac{1}{n}\sum_{i=1}^n (y_i-\hat y_i)^2. $$

In the case where the residuals $\varepsilon := y-\hat y$ have *mean zero*, minimising this loss function leads to predicting the **conditional mean**:

$$ \hat Y = \mathbb E[Y|X]. $$

That's well and good, but this says nothing about how varied the residuals are. We might even be in a situation where the variances of the individual residuals are different, a property called [heteroscedasticity](https://en.wikipedia.org/wiki/Heteroscedasticity).

**Quantile regression** is the process of changing the MSE loss function to one that predicts *conditional quantiles* rather than conditional means. Indeed, the "germ of the idea" in [Koenker & Bassett (1978)](https://www.jstor.org/stable/1913643) was to rephrase quantile estimation from a *sorting* problem to an *estimation* problem. Namely, for $q\in(0,1)$ we define the **check function**

$$ \rho_q(x) := \left\{\begin{align}{ll}x(q-1) & \text{if }x<0 \\ xq & \text{otherwise}\right. $$

We then define the associated **mean quantile loss** as

$$ \textsf{MQL}(y, \hat y) := \frac{1}{n}\sum_{i=1}^n \rho_q(y_i-\hat y_i). $$

To find out what the optimal estimate for this loss function is, we're trying to find $\hat y$ such that

$$ 
\begin{align}
  \mathbb E[\rho_q(Y-\hat y)] &= (q-1)\int_{-\infty}^{\hat y}f(t)(t-\hat y)dt + q\int_{\hat y}^\infty f(t)(t-\hat y)dt \\
  &= q\int f(t)(t-\hat y)dt - \int_{-\infty}^{\hat y}f(t)(t-\hat y)dt \\
  &= q\int f(t)tdt - \hat yq\int_{\mathbb R}f(t)dt - \int_{-\infty}^{\hat y}f(t)tdt + \hat y\int_{-\infty}^{\hat y}f(t)dt \\
  &= q\int f(t)tdt - \hat yq - \int_{-\infty}^{\hat y}f(t)tdt + \hat yF(\hat y) \\
\end{align}
$$

where $f$ is the [PDF](https://en.wikipedia.org/wiki/Probability_density_function) of Y. If we differentiate with respect to $\hat y$ and set the expression equal to zero then we get that

$$ 0 = -q - \hat yf(\hat y) + \hat yf(\hat y)
