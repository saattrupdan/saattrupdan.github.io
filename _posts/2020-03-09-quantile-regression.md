---
layout: post
mathjax: true
title: Quantile regression
meta-description: Introduction to quantile regression for prediction intervals, a discussion of the pros and cons, and implementations for both linear quantile regression and quantile neural networks in PyTorch.
---

When we are performing regression analys using complicated predictive models such as neural networks, knowing how *certain* the model is is highly valuable in many cases, for instance when the applications are within the health sector. The [bootstrap prediction intervals](https://saattrupdan.github.io/2020-03-01-bootstrap-prediction/) that we covered last time requires us to train the model on a large number of bootstrapped samples, which is unfeasible if training the model takes many hours or days, leaving us stranded. 

Thankfully, there are alternatives. One of those is *quantile regression*, which we'll have a closer look at in this post.

This post is part of my series on quantifying uncertainty:
  1. [Confidence intervals](https://saattrupdan.github.io/2020-02-20-confidence/)
  2. [Parametric prediction intervals](https://saattrupdan.github.io/2020-02-26-parametric-prediction/)
  3. [Bootstrap prediction intervals](https://saattrupdan.github.io/2020-03-01-bootstrap-prediction/)
  4. Quantile regression


## Going beyond the mean
When we are fitting predictive models for regression we tend to use the **mean squared error (MSE)**, or **l2**, loss function

$$ \textsf{MSE}(y, \hat y) := \frac{1}{n}\sum_{i=1}^n (y_i-\hat y_i)^2. $$

In the case where the residuals $\varepsilon := y-\hat y$ have *mean zero*, minimising this loss function leads to predicting the **conditional mean** $\hat Y = \mathbb E[Y\mid X]$. That's well and good, but this says nothing about how varied the residuals are. We might even be in a situation where the variances of the individual residuals are different, a property called [heteroscedasticity](https://en.wikipedia.org/wiki/Heteroscedasticity).

**Quantile regression** is the process of changing the MSE loss function to one that predicts *conditional quantiles* rather than conditional means. Indeed, the "germ of the idea" in [Koenker & Bassett (1978)](https://www.jstor.org/stable/1913643) was to rephrase quantile estimation from a *sorting* problem to an *estimation* problem. Namely, for $q\in(0,1)$ we define the **check function**

$$ \rho_q(x) := \left\{\begin{array}{ll}x(q-1) & \text{if }x<0 \\ xq & \text{otherwise}\end{array}\right. $$

We then define the associated **mean quantile loss** as

$$ \textsf{MQL}(y, \hat y) := \frac{1}{n}\sum_{i=1}^n \rho_q(y_i-\hat y_i). $$

Let's check that the optimal estimate for this loss function is actually the quantiles. We're trying to find $\hat y$ that minimises

$$ 
\begin{align}
  \mathbb E[\rho_q(Y-\hat y)] &= (q-1)\int_{-\infty}^{\hat y}f(t)(t-\hat y)dt + q\int_{\hat y}^\infty f(t)(t-\hat y)dt \\
  &= q\int f(t)(t-\hat y)dt - \int_{-\infty}^{\hat y}f(t)(t-\hat y)dt \\
  &= q\int f(t)tdt - \hat yq\int f(t)dt - \int_{-\infty}^{\hat y}f(t)tdt + \hat y\int_{-\infty}^{\hat y}f(t)dt \\
  &= q\int f(t)tdt - \hat yq - \int_{-\infty}^{\hat y}f(t)tdt + \hat yF(\hat y)
\end{align}
$$

where $f$ is the [PDF](https://en.wikipedia.org/wiki/Probability_density_function) of Y. If we differentiate with respect to $\hat y$ and set the expression equal to zero then we get that

$$ 0 = -q - \hat yf(\hat y) + F(\hat y) - \hat yf(\hat y) = F(\hat y) - q, $$

showing that $\hat y \in F^{-1}[q]$, i.e. that it *is* indeed a $q$'th quantile. This shows that the estimator we get from minimising the mean quantile loss is [unbiased](https://en.wikipedia.org/wiki/Bias_of_an_estimator). We can also apply the [weak law of large numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers) here to see that the estimator is [consistent](https://en.wikipedia.org/wiki/Consistent_estimator), as the quantile loss will converge in probability to $\mathbb E[\rho_q(Y-\hat y)]$ as $n\to\infty$, which we showed above means that $\hat y$ is the $q$'th quantile.


## Discussion: strengths and weaknesses
A clear strength of quantile regression, compared to the [bootstrap approaches to prediction intervals](https://saattrupdan.github.io/2020-03-01-bootstrap-prediction/), is that we only have to fit the model just once, by modyfying the model to output two extra values. This is incredibly useful when our model takes a long time to train, such as deep neural nets, where bootstrapping 1000 times is simply not computationally feasible.

A very neat side effect of quantile regression is that it can take of [heteroscedasticity](https://en.wikipedia.org/wiki/Heteroscedasticity) out of the box (simply because our models are not simply outputting constants), a feature that the bootstrap approaches failed to achieve (see the simulations below). Attempts have been made to make the bootstrap approach account for this, e.g. using the [wild bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29#Wild_bootstrap), but this is unsuitable for new predictions as it requires knowledge of the residuals.

One notable weakness of the quantile prediction intervals is that **the model is quantifying its own uncertainty**. This means that we are reliant on the model being able to correctly fit the data, so in a case where the conditional means of the data follow a linear trend but the quantiles don't, we would then have to choose a non-linear model to get correct prediction intervals. Further, if we're overfitting the training data then the prediction intervals will also become overfitted. 

We can remedy the latter by creating [confidence intervals](https://saattrupdan.github.io/2020-02-20-confidence/) around the quantile predictions, but then we're back at either the homoscedasticity scenario if we choose to create parametric confidence intervals, or otherwise we have to bootstrap again, losing what I think is the primary benefit of the quantile approach for prediction intervals.

In short, I'd personally use quantile regression when dealing with heteroscedastic data (with confidence intervals included if bootstrapping is feasible), or when dealing with an accurate predictive model that takes a long time to train, such as neural nets.


## Implementations and simulations
Let's start with a simple linear example, where we sample $n=1000$ training data points $X\sim\textsf{Unif}(0,5)$ and define our response variable as $Y = 5X - 5 + \varepsilon$ with $\varepsilon\sim\mathcal N(0,1)$. For the testing data I've chosen $100$ equidistributed points in the interval $[0,5]$. Implementing linear quantile regression is simple using the [statsmodels](https://www.statsmodels.org/stable/index.html) module:

```python
from statsmodels.regression.quantile_regression import QuantReg

# Set up the model - note here that we have to manually add the constant
# to the data to fit an intercept, and that `statsmodels` models needs 
# the training data at initialisation
cnst = np.ones((X_train.shape[0], 1))
qreg_model = QuantReg(Y_train, np.concatenate([cnst, X_train], axis = 1))

# Fit the linear quantile regression three times, for the interval 
# boundaries and the median
preds = []
for q in [0.05, 0.5, 0.95]:
    res = qreg_model.fit(q)
    intercept, slope = res.params[0], res.params[1]
    preds.append(slope * X_test + intercept)

prediction_intervals = np.stack(preds, axis = 1)
```

If we plot the residuals and the intervals we get the following, with 87% covering! This is all predicted in one shot, taking only ~0.08 seconds, compared to a bootstrap approach with only 100 resamples taking roughly four times as long.

![Plot of quantile linear regression prediction interval, where the interval encloses most of the residuals](/img/quantile-linear-regression.png)

The most interesting use case of quantile regression to me is in conjunction with neural networks, so let's see how we could implement that in `PyTorch`. We'd need to be able to modify any network to predict the two extra values corresponding to the relevant quantiles, and implement the quantile loss. 

Let's start with the wrapper. The following module duplicates the model three times using Python's built-in `copy` module, and the two extra modules are then predicting the *offset* from the prediction to the quantile. I've done it this way to ensure that the lower predicted quantile is guaranteed to always be below the prediction, which is again always below the upper predicted quantile. 

I'm squaring the outputs of these lower and upper quantile predictions for the same reason. We could just take the absolute value here, but differentiable operations make gradient descent more stable.

```python
class QuantileRegressor(nn.Module):
    def __init__(self, model):
        super().__init__()
        import copy
        self.model = model
        self.lower = copy.deepcopy(model)
        self.upper = copy.deepcopy(model)
        
    def forward(self, x):
        preds = self.model(x)
        lower = preds - self.lower(x) ** 2
        upper = preds + self.upper(x) ** 2
        return torch.cat([lower, preds, upper], dim = 1)
```

The loss function then computes the mean of the three quantile losses: the two quantiles and the median. I've used ReLU functions here to implement the control statement in the definition of the check function. If we would like a smoother version we could replace the ReLU's by its smoother variants such as the [GELUs](https://arxiv.org/abs/1606.08415). Note that, in the special case of the median, the quantile loss simply reduces to the absolute error.

```python
class QuantileLoss(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.lower, self.upper = ((1 - alpha) / 2, (1 + alpha) / 2)
        
    def forward(self, preds, target):
        residuals = target.unsqueeze(1) - preds
        lower_loss = torch.mean(F.relu(residuals[:, 0]) * self.lower - 
          F.relu(-residuals[:, 0]) * (self.lower - 1))
        median_loss = torch.mean(torch.abs(residuals[:, 1]))
        upper_loss = torch.mean(F.relu(residuals[:, 2]) * self.upper - 
          F.relu(-residuals[:, 2]) * (self.upper - 1))
        return (lower_loss + median_loss + upper_loss) / 3
```

Now, with these tools at hand, I've trained an [MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron) with hidden layers $[256, 256, 256]$ and [GELU](https://arxiv.org/abs/1606.08415) activations to achieve the following, which is basically the same thing as in the linear case, here with 86% covering.

![Plot of quantile MLP prediction interval, where the interval encloses most of the residuals](/img/quantile-linear-mlp.png)

To see an example of how the quantile approach deals with heteroscedasticity, let's multiply our noise terms $\varepsilon\sim\mathcal N(0,1)$ with our independent $X$, so that the observations become more noisy over time. Below we see that the quantile approaches really shine when compared to the [bootstrap approaches](https://saattrupdan.github.io/2020-03-01-bootstrap-prediction/):

![Residual- and prediction interval plots for quantile regression and quantile MLP, and bootstrap prediction intervals for linear regression and random forests. The data has no variance to start with but becomes very noisy towards the end, and the quantile intervals pick this up by starting off being very narrow and then slowly becomes wider, where the bootstrap intervals are of nearly constant width throughout](/img/quantile-linear-heteroscedastic.png)

Here the quantile intervals have a coverage of 88% and 87%, and the bootstrapped intervals cover 88% and 89%. So they both perform roughly as they should, but here the average interval length in the quantile case is ~8.1, which is ~9.9 for the bootstrap intervals. From the plots we see that the quantile intervals capture the actual variance a lot better.

To see what happens in the non-linear case, we now take as explanatory variables $\vec X=(X_1, X_2, X_3, X_4, X_5)$ following a multivariate normal distribution with means $\mu_i\sim\textsf{Unif}(-1, 1)$ and covariances $\Sigma_{ij}\sim\textsf{Unif}(-2, 2)$, and define our response variable as

$$ Y := \exp(X_1) + X_2X_3^2 + \log(|X_4 + X_5|) + \varepsilon $$

with $\varepsilon\sim\mathcal N(0,1)$. We again sample $n=1000$ data points for the training set and for our test set we sample 100 points uniformly from $[-4, 4]^5$. We then get the following, where the $x$-axis is the indices of the 100 test samples.

![Residual- and prediction interval plots for quantile regression and quantile MLP, and bootstrap prediction intervals for linear regression and random forests. Here the two linear regression variants are basically identical, the MLP intervals are really narrow and the residuals are small, and the random forest's residuals vary a bit more but with corresponding intervals being a lot narrower than the linear regression ones](/img/quantile-non-linear.png)

Starting from top-left and proceeding in Western-style reading order, we get coverages 43%, 41%, 59% and 61%. All quite far from the intended 90%, but we see at least that the quantile approach and the bootstrap approach yields roughly the same coverage. We see that the MLP fits the data really well, and with correspondingly narrower prediction intervals. Again, I emphasise that the neat feature here is that we have only trained the neural network *once*.

I'm hiding some detail in the above, as it was quite easy to mess up the MLP prediction intervals. To see why this is the case, simply note that since the model is treating the intervals just like any other prediction, it can **overfit the prediction intervals**. To see what could go wrong, let's take the same network and simply double the neurons in the three hidden layers.

![Residual- and prediction interval plot for the overfitting MLP. The residuals are much larger in magnitude, and the intervals are incredibly narrow around each residual.](/img/quantile-non-linear-overfit.png)

It looks a bit strange, which is because the intervals have nearly collapsed to zero length, giving a coverage of only 9%. This shows that we have to be really careful when employing this method with neural networks, and only believe the uncertainty estimates when we are sure that the model is not overfitting the data.
