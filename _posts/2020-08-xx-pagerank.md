---
layout: post
mathjax: true
title: The PageRank Algorithm
meta-description: 
---

I've recently started working with graph structures in the context of machine learning, and have found that I've opened what seems to be a reverse Pandora's box, full of neat algorithms that can pull out a lot of insights from graph structures. As a way of cementing my knowledge and hopefully also giving a different perspective, I'll do a series of blog posts of various graph algorithms that I find interesting and/or useful. I'll aim to cover both the theoretical foundation of the algorithms as well as concrete implementations and examples of them.

The algorithm I'd like to start with today is somewhat of a classic by now: Google's PageRank algorithm, developed in 1996 and originally designed to order search results, but which can be applied to any graph structure to get an idea of the most *important* nodes in the graph (where important here means most connected). There are two different versions of the algorithm: a global and a local one. They are very similar, but are used in completely different contexts. Let's get started.


## Some Intuition: The Random Web Surfer

Before we dive into the Mathematics of the algorithm, I'd like to start with an intuitive idea that will guide us along the way. We imagine a person who surfs around the web, clicking on a random link on every page. Every once in while the surfer ignores the current website however, and instead goes to a completely random site.

We then ask ourselves: how much of the web surfer's time will be spent at each individual website? The algorithm in its essence is quite simple: we simply let the person surf around and record how often they visit each node. The fact that this procedure will eventually terminate is then the crucial result that makes the algorithm useful.

[gfx]


## Markov Chains

The Mathematics around the PageRank algorithm mostly concerns Markov chains, so let's start with a definition.

> **Definition**. A **Markov chain** is a sequence $(X_n)_{n\in 0}^\infty$ of random variables $X_n$ for some *state space* $S\subseteq\mathbb N_0$, such that 
>   1. (Markov property) $P(X_{n+1} | X_0,\dots,X_n) = P(X_{n+1} | X_n)$; and
>   2. (Time homogeneity) $P(X_{n+1} = s | X_n = t)$ is independent of $n$, for every $s, t\in S$.

[intuition]

[example]

[homogeneity enables a transition matrix]

> **Definition**. A *stationary distribution* of a Markov chain $(X_n)$ is a discrete distribution $\pi:S\to\mathbb R$ such that $\pi T = \pi$, where $T$ is the transition matrix.

[example]

[theorem of existence of stationary distribution]


## The Algorithm

[definition of pagerank]

[example]


## Python Implementation

[implementation using networkx]
[implementation using neo4j]
