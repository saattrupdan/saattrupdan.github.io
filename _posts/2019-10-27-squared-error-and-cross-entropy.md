---
layout: post
mathjax: true
title: Squared Error and Cross Entropy
---

When introduced to machine learning, practically oriented textbooks and online courses focus on two major loss functions, the *squared error* for regression tasks and *cross entropy* for classification tasks, usually with no justification for *why* these two are important. I'll here show that they're both instances of the same concept: maximum likelihood estimation.


## Revisiting old friends

Before we dive into why we might be interested in these loss functions, let's ensure that we're on the same page and quickly recall how they are defined. Whenever we have two vectors $x,y\in\mathbb R^n$, the **mean squared error** is given as

$$ \frac{1}{n}\sum_{k=1}^n(x_k-y_k)^2, $$

and, assuming now that $x\in\{0,1\}^n$ and $y\in(0,1]^n$, the **binary cross-entropy** is defined as

$$ -\frac{1}{n}\sum_{k=1}^n (x_k\log y_k + (1-x_k)\log(1-y_k))). $$

In [Scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) we find these as `sklearn.metrics.mean_squared_error` and `sklearn.metrics.log_loss`, in [Keras](https://keras.io/losses/) as `keras.losses.mean_squared_error` and `keras.binary_crossentropy`, and in [PyTorch](https://pytorch.org/docs/stable/nn.html#loss-functions) as `torch.nn.MSELoss` and `torch.nn.BCELoss`.


## A statistical point of view

To explain why these two losses achieve what we want, we first need to agree on what *exactly* it is that we want to achieve. Let's consider a running regression example. In this case we're trying to estimate the value of a variable, which for instance could be the number of active Twitter users worldwide in given quarter:

![alt](/img/twitter.png)

We assume here that there is a *true* answer, meaning that there is a distribution which will accurately model the number of Twitter users throughout all time. We only know these true values for historic data and we'd like to be able to predict the future values.

Let us write $X$ for the random variable following the true distribution, meaning that $X(q)$ will be the actual number of Twitter users in a given quarter $q$. Say that we have chosen our favourite model $\hat X_{\theta}$ with parameter $\theta$, which attempts to model the historic data we have available. 


## Mean squared error

We will now assume that the errors our model is making is *normally distributed* around the true values. This assumes that our model is a reasonable choice, of course. If our model simply output the value $1$ for all quarters then this would be far from true, so we're simply excluding that possibility.

If we further assume that our observations are independently drawn from this distribution, then this means that we can write the density function of $\hat X_{\theta}$'s distribution as

$$ \prod_{q=1}^{37}\mathcal{N}(X(q),\sigma^2) = \prod_{q=1}^{37}\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{1}{2\sigma^2}(\hat X_{\theta}(q)-X(q))^2\right). $$

By finding the value for our model parameter $\theta$ which maximises this value we're thus effectively maximising the probability of our model being correct, which is what we want! This is called **maximum likelihood estimation**, where the above expression is our likelihood function in this case.

The exponentials are a bit annoying to deal with, so instead of dealing with the above expression we'll apply a logarithmic transform and we wind up with the **log-likelihood function**

$$ \sum_{q=1}^{37}\frac{1}{\sqrt{2\pi\sigma^2}}-\frac{1}{2\sigma^2}(\hat X_{\theta}(q)-X(q))^2. $$

As the logarithm is strictly increasing our job is equivalent to maximising this new expression, and as we're maximising with respect to $\theta$ we can ignore the constants. If we also flip the sign then we're left with the job to *minimise*

$$ \sum_{q=1}^{37}(\hat X_{\theta}(q)-X(q))^2, $$

which is exactly the squared error! This shows that 

<center>
  <i>Minimising mean squared error is equivalent to maximising likelihood, assuming that our errors are normally distributed.</i>
</center>


## Cross entropy

The story about cross entropy turns out to be surprisingly similar to the squared error. To see this, recall the notion of [expected value](https://en.wikipedia.org/wiki/Expected_value) of a random variable:

$$ EX := \int xp(x)d\mu, $$

where $p(x)$ is the density function of $X$'s distribution. If we assume that our probability space is $\{1,2,\dots,n\}$ with $\mu$ being the discrete probability measure, then the expectation is simply the average:

$$ EX = \sum_{k=1}^n p(k)X(k) = \frac{1}{n}\sum_{k=1}^nX(k). $$

With this is mind, we can rewrite the mean squared error as $E[(X-\hat X_\theta)^2]$, i.e. when we're minimising the squared error then we're reducing the expected value of the squared distance between our predictions and the true values.

Here I was taking $\mu$ to be the discrete probability measure, giving equal probabilities to all the $n$ outcomes. In the mean squared case this translates to giving equal weight to the squared differences throughout all quarters: whether our prediction is wrong in the beginning of 2015 or end of 2018 doesn't really matter to us.

But what if we're dealing with a different distribution? When we're dealing with classifications our true variable $X$ follows a probability distribution: If, say, 25% of the values are true and the rest false, then $X$ follows the distribution with density function 

$$ p(k) = \left\{\begin{array}{ll}\tfrac{4}{n} & \text{if the $k$'th observation is true}\\ 0 & \text{otherwise}\end{array}\right. $$

and $1-X$ then follows the distribution with density function $1-p$. Using these facts, and letting $\text{supp}(X)$ stand for the **support** of $X$, meaning the values for which it is non-zero, we can now rewrite the cross entropy as

$$
\begin{align}
&-\frac{1}{n}\sum_{k=1}^n (X\log\hat{X}_\theta + (1-X)\log(1-\hat{X}_\theta))\\
=& -\frac{1}{|\text{supp}(X)|}\sum_{k=1}^n X\log\hat{X}_\theta - \frac{1}{|\text{supp}(1-X)|}\sum_{k=1}^n(1-X)\log(1-\hat{X}_\theta)\\
=& -\frac{4}{n}\sum_{k=1}^n X\log\hat{X}_\theta - \frac{4}{3n}\sum_{k=1}^n (1-X)\log(1-\hat{X}_\theta)\\
=& -E[\log\hat{X}_\theta] - E[\log(1-\hat{X}_\theta)], 
\end{align}
$$

which is precisely the two (negative) log-likehood functions corresponding to our two classes! This means that, just as before, *minimising* the cross entropy between the true variable $X$ and the predicted variable $\hat X_\theta$ is equivalent to *maximising* the likelihood that the predicted variable follow the two distributions of the true variable, corresponding to the two classes. The case with more than two classes is completely analogous, we'll just end up with more distributions.

So there you go, it's really all about maximum likelihood estimation!
