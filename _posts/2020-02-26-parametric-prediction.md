---
layout: post
mathjax: true
title: Parametric prediction intervals
meta-description: Parametric prediction intervals are produced for machine learning models using normal theory, which are different to the confidence intervals. As examples we consider the classical linear regression model with additive noise and show that the intervals work as intended in that case. However, as soon as we start to overfit the intervals get too narrow.
---

One aspect of machine learning that does not seem to attract much attention is quantifying the *uncertainty* of our models' predictions. In classification tasks we can *partially* remedy this by outputting conditional probabilities rather than boolean values, but what if the model is outputting 52%? Is that a clear-cut positive outcome? When it comes to regression tasks it is even worse, as we simply output a number with no uncertainty attached to it. As we saw with [confidence intervals](https://saattrupdan.github.io/2020-02-20-confidence/), we can compute these intervals both parametrically using normal theory and unparametrically using bootstrapping methods.

This post is part of my series on quantifying uncertainty:
  1. [Confidence intervals](https://saattrupdan.github.io/2020-02-20-confidence/)
  2. Parametric prediction intervals
  3. [Bootstrap prediction intervals](https://saattrupdan.github.io/2020-03-01-bootstrap-prediction/)
  4. [Quantile regression](https://saattrupdan.github.io/2020-03-09-quantile-regression/)

Assuming we have a [univariate](https://en.wikipedia.org/wiki/Univariate) predictive model $\mu\colon\mathbb R^n\to\mathbb R$ trained on training data $\\\{(x_i,y_i)\in\mathbb R^{n+1}\mid i < n\\\}$, an **$\alpha$-prediction interval** for $\alpha\in(0,1)$ associated to a new sample $x_0$ is an interval $(a,b)\subset\mathbb R$ such that, if we were to continue sampling new training data, fit our model to the samples and produce new predictions for $x_0$, then the true value $y_0$ will land within $(100 * \alpha)$% of the intervals.


## Computing prediction intervals

Let's have a look at a simple example. Assume that we have data $\mathcal D \sim 3X - 5 + \varepsilon$ with $X\in\text{Unif}(0,1)$ and $\varepsilon\sim\mathcal N(0,\sigma^2)$, where $\text{Unif}(0,1)$ is the [uniform distribution](https://saattrupdan.github.io/2019-05-22-uniform/) and $\mathcal N(0,1)$ being the [normal distribution](https://saattrupdan.github.io/2019-06-05-normal/). Let's sample $N=50$ training samples from our data distribution and fit a linear regression model.

![Linear data with additive normal noise and a fitted linear regression line.](/img/prediction-data.png)

If we now were to sample $n=200$ equidistributed test samples from the same distribution, we *could* just supply the linear regression prediction at those points, but we see from the above plot that the true values corresponding to those test samples would probably not *exactly* equal the predicted values, so we'd like to quantify our uncertainty of our predictions. Let's say that we'd like to calculate a 90% prediction interval.

Note first that a 90% confidence interval would **not** work in this case, since such all such a confidence interval would show is how confident we are that our prediction is equal to the *mean* of potential predictions, and it doesn't take the variance into account. We also saw [last time](https://saattrupdan.github.io/2020-02-20-confidence/) that the length of a confidence interval tends to 0 as our sample size increase, which wouldn't make sense in a prediction scenario.

![The same data as before but with a way too narrow confidence interval.](/img/prediction-confidence.png)

A quick calculation shows that the **coverage**, being the proportion of the true test values that land within the interval, is only 20%, far from the desired 90%. Let's have a look at what's happening here. Under our model assumption we only have the $\varepsilon$ component as noise, so we're trying to quantify how these noise terms vary. We can estimate the distribution of the noise terms by computing the **residuals** $\varepsilon_i := y_i-\hat y_i$ for our training data (so $i = 0,\dots,N-1$).

![Plot of the sample residuals, which are roughly normally distributed.](/img/prediction-residuals.png)

Now, given a new test sample $x_0$, we would like to guess where the residual $\varepsilon_0$ associated to our prediction $\hat y_0$ might land. We're assuming that $\varepsilon\sim\mathcal N(0,\sigma^2)$ for some variance $\sigma^2$, and we've [previously seen](https://saattrupdan.github.io/2020-02-20-confidence/) that $\bar\varepsilon_i\sim\mathcal N(0,\tfrac{\sigma^2}{n})$. This means that

$$ \varepsilon_0 - \bar\varepsilon\sim\mathcal N(0, \sigma^2 + \tfrac{\sigma^2}{n}), $$

so that $\varepsilon_0\sim\mathcal N(\bar\varepsilon, \sigma^2 + \tfrac{\sigma^2}{n})$. As [we saw with confidence intervals](https://saattrupdan.github.io/2020-02-20-confidence/), we now use $s_N^2 := \tfrac{1}{n-1}\sum_{i=1}^N(\varepsilon_i - \bar\varepsilon)^2$ as our unbiased estimate of $\sigma^2$, and that $\tfrac{\varepsilon_0}{s_N} \sim T^{N-1}$, the [t-distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution) with $(N-1)$ degrees of freedom. Summa summarum, we get a 90%-prediction interval 

$$ \hat y_0 + \bar\varepsilon \pm F^{-1}(0.95)s_N(1 + \tfrac{1}{n}) $$

with $F$ being the CDF for the $t$-distribution with $(N-1)$ degrees of freedom.

![Plot of the prediction interval, nearly covering all the true values](/img/prediction-normal-pi.png)

The coverage in this case is very close to 90%. I repeated the experiment 10 times and got the following coverage values:

| Experiment | #0 | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 |
| Coverage   | .91 | .87 | .87 | .87 | .91 | .92 | .87 | .93 | .90 | .86 |

The values average to 88.75%, which is quite close to our desired 90%. After 10,000 repetitions they average to 89.2%.


## Where the parametric approach doesn't work

A standing assumption throughout the above method is that the *model assumption* is correct; i.e., that the true values are actually iid normally distributed around the predictions. In particular, the intervals would be identical for the training data and testing data. This is fine with the above linear regression example, but in practice we will often *overfit* the training set to some degree. This means that our intervals will be constructed with respect to the *training* error and not the *validation* error.

If we for instance simply switch out the linear regression model in the above example with a model that is often used in practice, a random forest, we get the following.

![Plot of the prediction interval around the random forest predictions, which is way too narrow](/img/prediction-random-forest.png)

Here the coverage is only 50%, quite far from the intended 90%. To go to an even more extreme case, this is what happens if we fit a single decision tree to the data.

![Plot of the prediction interval around the decision tree predictions, which has length zero](/img/prediction-decision-tree.png)

Here we see that the intervals have shrunk to nothing, giving a coverage of **0%**! We therefore see that we are heavily dependant on our model assumption in this parametric setting, but thankfully there are non-parametric methods as well which take care of this issue, which we will see in the next post in this series.
