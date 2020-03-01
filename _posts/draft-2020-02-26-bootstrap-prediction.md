---
layout: post
mathjax: true
title: Bootstrapping prediction intervals
meta-description: 
---

Continuing from [where we left off](https://saattrupdan.github.io/2020-02-26-parametric-prediction/), in this post I will discuss a general way of producing accurate prediction intervals for all machine learning models that are in use today. The algorithm for producing these intervals uses [bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29) and was introduced in [Kumar and Srivastava (2012)](https://ntrs.nasa.gov/search.jsp?R=20130014367).

This post is part of my series on quantifying uncertainty:
  1. [Confidence intervals](https://saattrupdan.github.io/2020-02-20-confidence/)
  2. [Parametric prediction intervals](https://saattrupdan.github.io/2020-02-26-parametric-prediction/)
  3. Bootstrap prediction intervals


## The setup
Let's say that we're working with a $d$-dimensional feature space and that we only have a single response variable. We will then assume that the true model $y\colon\mathbb R^d\to\mathbb R$ is of the form

$$ y(x) = \psi(x) + \varepsilon(x), $$

where $\psi\colon\mathbb R^d\to\mathbb R$ is the "main model function" and $\varepsilon\colon\mathbb R^d\to\mathbb R$ is a noise function. These will satisfy that
\begin{enumerate}
  \item $\psi$ should be **deterministic**, meaning that is has no random elements;
  \item $\psi$ is "sufficiently smooth";
  \item $\varepsilon(x)$ are iid for all $x\in\mathbb R^d$, with mean $0$ and finite variance.
\end{enumerate}

For a precise definition of "sufficiently smooth" check out the paper, but we note that a sufficient condition for satisfying this is to be [continuously differentiable](https://en.wikipedia.org/wiki/Differentiable_function#Differentiability_classes).

On top of the true model we of course also have our model estimate $\hat y_n\colon\mathbb R^d\to\mathbb R$, which has been trained on a sample of size $n$. We also assume a couple of things about this model:
\begin{enumerate}
  \item $\hat y_n$ is deterministic;
  \item $\hat y_n$ is continuous;
  \item $\hat y_n$ converges pointwise to some $\hat y\colon\mathbb R^d\to\mathbb R$ as $n\to\infty$;
  \item $\mathbb E[\hat y_n(x)-\psi(x)]^2\to 0$ as $n\to\infty$ for every $x\in\mathbb R^d$.
\end{enumerate}

Most notable is assumption $(4)$, stating that our model estimate $\hat y_n$ will estimate the true model $\psi$ *perfectly* as we gather more data. In other words, we're essentially assuming that we can get *zero training error*. This is fine for most unregularised models (not all though, with linear regression being an example), but as soon as we start regularising then this won't hold anymore.

It turns out that all of this is still fine if we simply remove $(4)$ and instead assume that $\eta(x):=\lim_{n\to\infty}\mathbb E[\hat y_n(x)-\psi(x)]^2$ exists for every $x\in\mathbb R^d$. This then corresponds to the **bias** of the model.

## Two types of noise
asd

## Prediction interval algorithm
asd

## Theoretical validity of intervals
asd

## Simulations
asd
