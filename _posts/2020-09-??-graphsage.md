---
layout: post
mathjax: true
title: GraphSAGE
meta-description: 
---

Graph representation learning is the process of learning latent dense representations/embeddings of nodes or subgraphs of a graph, which somehow describe the behaviour of the node or subgraph, and which can concretely be used as neat features to feed into our machine learning models.

Last time we had a look at the DeepWalk algorithm, which produces embeddings for every node in a graph. A crucial downside to that algorithm is that it does not take into account any *inherent* features that each node has. Say, if every node is a town, then we might want to include the population count, area and so on as we are computing the embeddings. 

Another downside to DeepWalk is that if a new node appears then we have no idea what embedding to assign it to, and we have to resort to training the embeddings from scratch, which can potentially be very time consuming.

In this post we'll talk about the **GraphSAGE** algorithm, which is a simple algorithm that neatly solves these two problems.

This post is part of my series on graph algorithms:
  1. [PageRank](https://saattrupdan.github.io/2020-08-07-pagerank/)
  2. [DeepWalk](https://saattrupdan.github.io/2020-08-24-deepwalk/)
  3. GraphSAGE


## Induction vs Transduction

-content-


## Sample

-content-


## AggreGatE

-content-


## Implementation

-content-
