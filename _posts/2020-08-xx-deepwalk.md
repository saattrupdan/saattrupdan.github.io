---
layout: post
mathjax: true
title: DeepWalk
meta-description: 
---

Deep learning has almost exclusively been working with simple objects: images and text. By simple I am here referring to the _graphical structure_ of these objects, where a word is a linear sequence of letters, a document is a linear sequence of words, and an image is a rectangular grid of pixels. **Graph Neural Networks (GNNs)** were invented in [Gori et al (2005)](https://doi.org/10.1109/IJCNN.2005.1555942) by researchers from Università di Siena in Italy, and are networks which process information without requiring the input to have a particular rigid structure. 

Recent work within GNNs has focused on the development of **representation learning** for graphs, and I'll be writing about a handful of these ideas, each superseding the former. We start with the **DeepWalk** algorithm, introduced in [Perozzi et al (2014)](https://doi.org/10.1145/2623330.2623732) by researchers from Stony Brook University in the USA.

This post is part of my series on graph algorithms:
  1. [PageRank](https://saattrupdan.github.io/2020-08-07-pagerank/)
  2. DeepWalk


## The SkipGram Algorithm

The DeepWalk algorithm is intimately connected to the SkipGram algorithm introduced a year before, in the Google paper [Mikolov et al (2013)](https://arxiv.org/abs/1301.3781). The goal of the SkipGram algorithm is to produce vector representations of words, solely from data. The fundamental idea in the SkipGram algorithm is that the model should attempt to predict the neighbouring words of a given input word. 

A crucial notion here is *neighbour*. In this algorithm, we denote an **n-neighbour** of a given word to be any word at most $n$ spaces away from the word. For example, in the sentence "We are learning about SkipGram", the 2-neighbours of "about" are "are", "learning" and "SkipGram". Note that we do not care *how* close the neighbours are, as we are not ranking them in any way. Here $n$ is a hyperparameter in the algorithm, for "neighbouring words" to have a precise meaning.

But where is the vector representation of the word then? Indeed, the missing piece to the SkipGram algorithm is that it consists of an **encoder** and a **decoder**, see Figure 1. These can in principle be arbitrary neural networks, and the output of the encoder-decoder model should then be a probability distribution of the neighbouring words of the input word. 

![A diagram of the SkipGram architecture, which takes an input word and attempts to predict one of the neighbouring words.](/img/skipgram.png)
*Figure 1. The SkipGram architecture, from the original paper.*

This way of designing the architecture means that we have an *intermediate representation*, namely the output of the encoder, which we can use as the representation of the input word after we have trained the model.

In the original implementation from the above paper by Mikolov et al, which they denoted [Word2Vec](https://en.wikipedia.org/wiki/Word2vec), simply used a linear projection for the encoder and another linear embedding for the decoder (no non-linearities used at all). This made it highly computationally efficient, making it possible to process millions of words in a reasonable amount of time.


## From SkipGram to DeepWalk

Knowing what the SkipGram algorithm is about, the leap to DeepWalk is not far. As I mentioned above, the context of SkipGram is a *linear* chain (of words), so when we're going from the linear context to an arbitrary graph structure, we only have to change the features in SkipGram which used the linearity, which was in the definition of neighbour.

In a general graph we *could* mimic the definition of SkipGram and simply define $n$-neighbours in the same way. Namely, a node which is at most $n$ hops away from the input node. The problem with this approach is that graphs are usually highly connected, so even going only 5 hops away from your node, you might suddenly have reached every node in the graph. As our graph might contain millions of nodes, this becomes computationally infeasible.

What DeepWalk does is to *sample* the neighbours in a particular way, rather than considering all of them at once. This is done through **random walks**, intuitively being finite sequences of nodes in the graph, obtained by starting from a random node and "walking" randomly around the graph. We can define this formally as a finite *Markov chain* with uniform transition probabilities; we defined these terms [last time](https://saattrupdan.github.io/2020-08-07-pagerank/).

What we then do is two-fold. We firstly do a random walk $w_x$ at every node $x$ in the graph. Then, for every random walk $w_x$ and every node $y\in w_x$, we can now consider the $n$-neighbours of $y$ to be the nodes at most $n$ hops away *within this random walk* $w_x$. This point is worth re-iterating: we are using the random walks to *reduce* the neighbour concept back to the linear case!

[pic]

This also means that in every epoch we will have started random walks at every node, and we will therefore have processed many nodes multiple times, as they could have occured in multiple random walks. Thus, an epoch here is slightly different from normal deep learning training loops.


## Implementations of DeepWalk

asd


## Final Comments

I mentioned in the introduction that this was the first (very successful) attempt at producing representations for nodes in a graph. Several other methods have superseded this algorithm by now, so this post is mainly to understand how this field of graph representation learning has progressed and to have some context for the representation algorithms that I will cover in future posts.
