---
layout: post
mathjax: true
title: Evaluating predictions
meta-description: Prediction intervals
---

One aspect of predictive modelling that does not seem to attract much attention is quantifying the *uncertainty* of our models' predictions. In classification tasks we can *partially* remedy this by outputting conditional probabilities rather than boolean values, but what if the model is outputting 52%? Is that a clear-cut positive outcome? When it comes to regression tasks it is even worse, as we simply output a number with no uncertainty attached to it. As we saw with [confidence intervals](https://saattrupdan.github.io/2020-02-20-confidence/), we can compute these intervals both parametrically using normal theory and unparametrically using bootstrapping methods.

This post is part of my series on quantifying uncertainty:
  1. [Confidence intervals](https://saattrupdan.github.io/2020-02-20-confidence/)
  2. Prediction intervals


## Prediction intervals

Assuming we have a [univariate](https://en.wikipedia.org/wiki/Univariate) predictive model $\mu\colon\mathbb R^n\to\mathbb R$ trained on training data $\{(x_i,y_i)\in\mathbb R^{n+1}\mid i < n\}$, an **$\alpha$-prediction interval** for $\alpha\in(0,1)$ associated to a new sample $x_0$ is an interval $(a,b)\mathbb R$ such that, if we were to continue sampling new training data, fit our model to the samples and produce new predictions for $x_0$, then the true value $y_0$ will land within $(a,b)$ in $(100 * \alpha)$% of the intervals.

Let's have a look at a simple example. Assume that we have data $\mathcal D \sim 3X - 5 + \varepsilon$ with $X\in\text{Unif}(0,1)$ and $\varepsilon\sim\mathcal N(0,\sigma^2)$, where $\text{Unif}(0,1)$ is the [uniform distribution](https://saattrupdan.github.io/2019-05-22-uniform/) and $\N(0,1)$ being the [normal distribution](https://saattrupdan.github.io/2019-06-05-normal/). Let's sample $N=50$ training samples from our data distribution and fit a linear regression model.

![Linear data with additive normal noise and a fitted linear regression line.](/img/prediction-data.png)

If we now were to sample $n=200$ equidistributed test samples from the same distribution, we *could* just supply the linear regression prediction at those points, but we see from the above plot that the true values corresponding to those test samples would probably not *exactly* equal the predicted values, so we'd like to quantify our uncertainty of our predictions. Let's say that we'd like to calculate a 90% prediction interval.

Note first that a 90% confidence interval would **not** work in this case, since such all such a confidence interval would show is how confident we are that our prediction is equal to the mean of potential predictions. This means that if the noise is symmetrical, which is the case here, then the confidence interval would be artificially narrow:

![The same data as before but with a way too narrow confidence interval.](/img/prediction-confidence.png)

A quick calculation shows that only *20%* of the true test values land within the interval. Let's have a look at what's happening here. Under our model assumption we only have the $\varepsilon$ component as noise, so we're trying to quantify how these noise terms vary. We can estimate the distribution of the noise terms by computing the **residuals** on our training data: $\varepsilon_i := y_i-\hat y_i$ for $i = 0,..49$. 

![Plot of the sample residuals, which are roughly normally distributed.](/img/prediction-residuals.png)

Now, given a new test sample $x_0$, we would like to guess where the residual $\varepsilon_0$ associated to our prediction $\hat y_0$ might land. We're assuming that $\varepsilon\sim\mathcal N(0,\sigma^2)$ for some variance $\sigma^2$, and we've [previously seen]() that $\bar\varepsilon_i\sim\mathcal N(0,\tfrac{\sigma^2}{n})$. This means that $\varepsilon_0\sim\mathcal N(\bar\varepsilon, \sigma^2 + \tfrac{\sigma^2}{n})$.

As [we saw with confidence intervals](), we now use $s_N^2 := \tfrac{1}{n-1}\sum_{i=1}^N(\varepsilon_i - \bar\varepsilon)^2$ as our unbiased estimate of $\sigma^2$, and that $\tfrac{\varepsilon_0}{s_N} \sim T^{N-1}$, the [t-distribution]() with $(N-1)$ degrees of freedom. Summa summarum, we get a 90%-prediction interval $\hat y_0 \pm T^{N-1}_{0.95}\bar\varepsilon$ with $T^{N-1}_{0.95}$ being the 95% quantile for the $t$-distribution with $(N-1)$ degrees of freedom.

![Plot of the prediction interval, nearly covering all the true values](/img/prediction-normal-pi.png)


