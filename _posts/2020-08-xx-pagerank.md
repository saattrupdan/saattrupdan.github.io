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

We then ask ourselves: how much of the web surfer's time will be spent at each individual website? The idea is that the more time is spent, the more important the website is. The algorithm in its essence is quite simple: we simply let the person surf around and record how often they visit each node. The fact that this procedure will eventually terminate is then the crucial result that makes the algorithm useful.

[gfx]

Of course, this simple procedure generalises beyond websites, and we can apply the algorithm whenever we're dealing with any kind of graph structure to get an idea of how *central* the nodes are.


## Markov Chains

The Mathematics around the PageRank algorithm mostly concerns Markov chains. Let's start with a formal definition and then dig into some intuition and examples.

> **Definition**. For a *state space* $S\subseteq\mathbb N_0$ we define a **Markov chain** (on $S$) as a sequence $(X_t)_{t\in 0}^\infty$ of random variables $X_t$, such that 
>   1. (Markov property) $P(X_{t+1} | X_0,\dots,X_t) = P(X_{t+1} | X_t)$; and
>   2. (Time homogeneity) $P(X_{t+1} = s' | X_t = s)$ is independent of $t$, for every $s, s'\in S$.

Here the intuition is that we imagine the Markov chain to be a random variable evolving through time. For instance, $X_t$ could be Chile's coffee consumption per capita in year $t$. For this example, the Markov property would translate to postulating that the amount of coffee drunk in Chile next year only depends upon the current year and is completely independent of how much coffee was enjoyed last year. Time homogeneity would say that if the same amount of coffee was had in two different years, then the coffee consumption for the following year in both cases follow the same distribution.

The reason why we care about the time homogeneity property is that it allows us to define for every pair of states $s,s'\in S$ the **transition probability** $p_{s,s'} := P(X_{t+1} = s' | X_t = s)$ for some, or equivalently any, $t\in\mathbb N_0$. These transition probabilities thus shows us that, given that Chile's coffee consumption is $s$ in a given year, what the probability distribution is for the coffee consumption in the following year. This comes with a neat graph structure:

[transition probability graph]

Here every state has a node (here we've just shown a few of them), and an arrow from state $s$ to $s'$ is labelled with the transition probability $p_{s,s'}$. To enable ease of notation we define the **transition matrix** $T$ as the $|S|\times|S|$ matrix with entries $T_{s,s'} := p_{s,s'}$. In the above example this takes the following form:

$$
T = 
$$

Now, with all this formalism we can now start to describe what we're trying to find. Namely, the **stationary distribution** associated to a Markov chain is a discrete distribution $\pi:S\to\mathbb R$ such that $\pi T = \pi$, where $T$ is the transition matrix and we abuse notation and view $\pi$ as a row vector of length $|S|$. Said in another way, $\pi(s)$ is equal to the sum of all the $\pi(s')$ which has an arrow going into $s$, weighted by their respective transition probabilities. This means that, roughly, the more in-links a state $s$ has, the higher $\pi(s)$ will be.

But how do we find the stationary distribution? This is where the following crucial theorem enters the picture. Intuitively, it says that by starting with a random choice of $\pi$ and updating $\pi$'s values by simply traversing the graph according to the transition probabilities, it will converge to the stationary distribution.

> **Theorem**. If a Markov chain has no loops and every state can be reached by every other state, then it has a unique stationary distribution. Further, if we let $\pi_0$ be the uniform distribution on the state space and define $\pi_{n+1} := \pi_n T$ with $T$ being the transition matrix, then the $\pi_n$'s converge to the stationary distribution.
[theorem of existence of stationary distribution]


## The Algorithm

[definition of pagerank]

[example]


## Python Implementation

[implementation using networkx]
[implementation using neo4j]
