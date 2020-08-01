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


## Markov Chains

asd


## The Algorithm

asd


## Python Implementation

asd
