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

In the case where the residuals $\varepsilon := y-\hat y$ have *mean zero*, minimising this loss function leads to predicting the **conditional mean** $\hat Y = \mathbb E[Y|X]$. That's well and good, but this says nothing about how varied the residuals are. We might even be in a situation where the variances of the individual residuals are different, a property called [heteroscedasticity](https://en.wikipedia.org/wiki/Heteroscedasticity).

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

$$ 0 = -q - \hat yf(\hat y) + F(\hat y) - \hat yf(\hat y) = F(\hat y) - q, $$

showing that $\hat y \in F^{-1}[q]$, i.e. that it *is* indeed the $q$'th quantile. This shows that the estimator we get from minimising the mean quantile loss is [unbiased](https://en.wikipedia.org/wiki/Bias_of_an_estimator). We can also apply the [weak law of large numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers) here to see that the estimator is [consistent](https://en.wikipedia.org/wiki/Consistent_estimator), as the quantile loss will converge in probability to $\mathbb E[\rho_q(Y-\hat y)]$ as $n\to\infty$, which we showed above means that $\hat y$ is the $q$'th quantile.


## Discussion: strengths and weaknesses
A clear strength of quantile regression, compared to the [bootstrap approaches to prediction intervals](https://saattrupdan.github.io/2020-03-01-bootstrap-prediction/), is that we only have to fit the model just once, by modyfying the model to output two extra values. This is incredibly useful when our model takes a long time to train, such as deep neural nets, where bootstrapping 1000 times is simply not computationally feasible.

A very neat side effect of quantile regression is that it can take of [heteroscedasticity](https://en.wikipedia.org/wiki/Heteroscedasticity) out of the box, a feature that the bootstrap approaches failed to achieve (see the simulations below). Attempts have been made to make the bootstrap approach account for this, e.g. using the [wild bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29#Wild_bootstrap), but this is unsuitable for new predictions as it requires knowledge of the residuals.

One notable difference between the quantile prediction intervals and the prediction intervals we've previously computed is that **the model is quantifying its own uncertainty** in the quantile case. This means that we are reliant on the model being able to correctly fit the data, so in a case where the conditional means of the data follow a linear trend but the quantiles don't, we would then have to choose a non-linear model to get correct prediction intervals. We can remedy this by creating [confidence intervals]() around the quantile predictions, but then we're back at either the homoscedasticity scenario if we choose to create parametric confidence intervals, or otherwise we have to bootstrap again, losing what I think is the primary benefit of the quantile approach for prediction intervals.

In short, I'd personally use quantile regression when I'm dealing with heteroscedastic data (with confidence intervals included if bootstrapping is feasible), or when I'm dealing with a strong predictive model that's hard to train, such as neural nets.


## Simulations
Comparison between quantile regression prediction intervals and bootstrap intervals.
  1. Linear regression on.. 
    1. Linear data with normal noise
    2. Linear data with heteroscedastic normal noise
    3. Linear data with asymmetric noise
    4. Non-linear data?
  2. Overfitting case: decision tree
  3. Neural network


## Variations
Smooth quantile regression, more?
