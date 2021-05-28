---
layout: post
mathjax: true
title: Graph Convolutional Neural Networks
meta-description: This is an introduction to graph convolutional neural networks, also called GCNs. These are approximations of spectral graph convolutions, which are defined using the graph Fourier transform, an analogue of the regular Fourier transform to the graph domain. Aside from going over the theoretical justification for GCNs, I also include some sample code in both PyTorch Geometric and Deep Graph Library (DGL).
---

As more and more businesses strive toward becoming data-driven, the use of
graph methods for storing relational data has been on the rise (
[1](https://www.forbes.com/sites/cognitiveworld/2019/07/18/graph-databases-go-mainstream/?sh=6f97faea179d),
[2](https://www.business-of-data.com/articles/graph-databases),
[3](https://www.eweek.com/database/why-experts-see-graph-databases-headed-to-mainstream-use/)).
Along with these graph databases comes more opportunities for analysing the
data, including the use of predictive machine learning models on graphs.

The current machine learning models currently used to model graphs are all
variants of the so-called *graph convolutional neural network*, abbreviated
GCNs, so covering that seems a good place to start!

This blog post grew out of my preparation for a London PyTorch MeetUp
presentation I gave last year. You can find my slides from this talk
[here](https://github.com/saattrupdan/talks/blob/master/pytorch-meetup-presentation/presentation.pdf).

This post is part of my series on graph algorithms:
  1. [PageRank](https://saattrupdan.github.io/2020-08-07-pagerank/)
  2. [DeepWalk](https://saattrupdan.github.io/2020-08-24-deepwalk/)
  3. Graph Convolutional Neural Networks


## A Recap on Convolutional Neural Networks

As the name suggests, the graph **convolutional** neural networks are related
to convolutional neural networks (CNNs). This connection turns out to be a bit
contrived, but we will get to that later. In any case, to make the analogue
clear, let's briefly go over how CNNs work.

The purpose of a CNN, say for image classification, is to learn how to
aggregate neighboring pixels. For every pixel we would like to train a function
$k$, that takes in the chosen pixel and all of its neighboring pixels as inputs
and uses that to come up with a representation for the chosen pixel. If we
further make the assumption that $k$ is a *linear* function, we can represent
it as a 3x3 matrix:

$$
k(x) = \left[\begin{array}{ccc} a&b&c\\ d&e&f\\ g&h&i \end{array}\right]x
$$

This matrix is called a **kernel**, or a **filter**. We can now perform the
[convolution](https://en.wikipedia.org/wiki/Convolution#Discrete_convolution)
of the pixels with this kernel as follows, to update the pixel values:

$$
(\textsf{pixels}\star k)_{m, n} :=
    \sum_{i=-1}^1\sum_{j=-1}^1 k_{i,j}\textsf{pixels}_{m-i,n-j}
$$

This of course assumes that all pixels have neighbouring pixels in all
directions, so for this to work properly we pad the edges of the image with an
extra pixel. Alternatively, we could choose to simply not update the edge
pixels, which would make the resulting representation be a bit smaller than the
original image.

Instead of training a simple filter, we normally train many filters at the same
time, so that they can learn different aspects of the image. One filter could
learn to recognise vertical lines, another one white space, and so on.

Now, why doesn't this work for graphs? The reason is that graphs don't have the
same kind of neat grid-like structure as images, which in particular means that
nodes in a graph can have wildly different numbers of neighbours. As a matrix
is of a fixed size, it simply cannot adapt to arbitrary graphs.


## Rephrasing the Problem in the Fourier Domain

The first step to move toward a more general convolution operator is to look at
the conventional convolution operator from a different perspective.

We can map any (measurable) function $f\colon\mathbb R^n\to\mathbb R$ to its
[multi-dimensional discrete-time Fourier
transform](https://en.wikipedia.org/wiki/Multidimensional_transform#Multidimensional_Fourier_transform)
$\textsf{fourier}(f)\colon\mathbb R^n\to\mathbb C$, given as:

$$
\textsf{fourier}(f)(\xi) :=
\sum_{k_1=-\infty}^\infty\sum_{k_2=-\infty}^\infty\cdots\sum_{k_n=-\infty}^\infty f(k_1, k_2, \dots, k_n)e^{-i\xi_1k_1-i-xi_2k_2-\cdots-i\xi_nk_n}
$$

Now, the reason why we care about this transformation, is the following
*Convolution Theorem*, stating that the convolution operation can be
rephrased as simple multiplication in the Fourier domain!

> **The Convolution Theorem.** For any two measurable functions
> $f,g\colon\mathbb R^n\to\mathbb R$ it holds that
> $$ \textsf{fourier}(f\star g) = \textsf{fourier}(f)\textsf{fourier}(g) $$

Now, how does this help us to generalise the convolution operation to general
graphs? What we've done here is moved from *shift operations* to *Fourier
transforms*, so the next step is generalising the regular Fourier transform to
general graphs.


## From Fourier to Graph Fourier

It turns out that there *is* an analogue of the Fourier transform to general
graphs. We have to go through yet another couple of hoops, however. First, for
a connected graph $\mathcal G$ with adjacency matrix $A$ we define the **graph
Laplacian** $L := D-A$, where $D$ is the diagonal degree matrix of $\mathcal
G$.

We'd like to compute the eigenvectors of the Laplacian, so we first ensure that
it's symmetric. This leads to the following **normalised graph Laplacian**:

$$
\hat L := D^{-\tfrac{1}{2}}LD^{-\tfrac{1}{2}} = I_N - D^{-\tfrac{1}{2}}AD^{-\tfrac{1}{2}},
$$

which *is* symmetric, by construction, so it has a complete set of orthonormal
eigenvectors $e_1,\dots,e_N$, where $N$ is the number of nodes in $\mathcal G$.
Letting $\lambda_1,\dots,\lambda_N$ be the associated eigenvalues, the **graph
Fourier transform** is then the function induced by:

$$
\textsf{graphFourier}(f)(\lambda_l) := \sum_{n=1}^N f(n)e_l^*(n),
$$

where $(-)^*$ is the complex conjugate. Note here that $f\colon\mathbb
R^N\to\mathbb R$ and $\textsf{graphFourier}(f)\colon\{\lambda_l\mid
l=1,\dots,N\}\to\mathbb C$.


## Spectral Graph Convolutions

Now, with the graph Fourier transform in place, we can then take inspiration
from the Convolution Theorem above and define the **spectral graph
convolution** between two functions $f,g\colon V\to\mathbb R$, with $V$ being
the set of vertices of $\mathcal G$, as

$$
f\star g := \textsf{graphFourier}^{-1}(\textsf{graphFourier}(f)\textsf{graphFourier}(g)).
$$

We simply pretend that the same relationship between convolutions and "Fourier
products" also holds in the graph domain, and define the convolution from that
relationship.


## Graph Convolutional Neural Networks

We *could* just stop here, and simply use the spectral graph convolutions. The
problem is that it's incredibly computationally expensive, as the graph Fourier
transform is $O(N^2)$, so the final job is about approximating this as best as
possible.

In [Hammond et al.
(2011)](https://www.sciencedirect.com/science/article/pii/S1063520310000552) it
was suggested that the spectral graph convolution could be approximated using
the so-called [Chebyshev
polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials), $T_n$, which
are given as $T_0(x) = 1$, $T_1(x) := x$ and $T_{n+1}(x) :=
2xT_n(x)-T_{n-1}(x)$. The $K$'th approximation then looks like

$$
f\star g \approx \sum_{k=0}^K f(k)T_k(\tilde L)g,
$$

where $\tilde L := \tfrac{2}{\lambda_{\text{max}}}\hat L-I_N$ with $I_N$ being
the $N\times N$ identity matrix and $\lambda_{\text{max}}$ being the largest
eigenvalue of $\hat L$.

In [Kipf and Welling (2017)](https://arxiv.org/abs/1609.02907), the paper where
GCNs were introduced, they make further approximations. Let's put our GCN hat
on, so that $f$ is now the kernel and $g$ is our node feature matrix, so let's
rename $f$ to $k$, and rename $g$ to $\textsf{nodeFeatures}$.

The first approximation they make is setting the Chebyshev approximation level
$K$ to $1$, reducing the approximation of $k\star\textsf{nodeFeatures}$ to
$k_0\textsf{nodeFeatures} + k_1\tilde L\textsf{nodeFeatures}$.

The second approximation is setting $\lambda_{\text{max}} = 2$, reducing it
further to $k_0g + k_1(\hat L - I_N)\textsf{nodeFeatures}$.

The third and last approximation they make is assuming that $k_0 = k_1$,
resulting in the final approximation

$$
f\star g \approx k_0(I_N + D^{-\tfrac{1}{2}}AD^{-\tfrac{1}{2}})\textsf{nodeFeatures}.
$$

Are we done yet? Not quite, there is one last problem we need to deal with.
$I_N + D^{-\tfrac{1}{2}}AD^{-\tfrac{1}{2}}$ now has eigenvalues in the range
$[0,2]$, so to avoid vanishing and exploding gradients, we normalise it: we set
$\tilde A := A + I_N$, $\tilde D_{ii} := \sum_j \tilde A_{ij}$. Finally, we end
up with the approximation:

$$
f\star g \approx k_0(\tilde D^{-\tfrac{1}{2}}\tilde A\tilde D^{-\tfrac{1}{2}})\textsf{nodeFeatures}.
$$

And success, there's our convolution!


## What does it all mean?

Phew, that was a lot. Let's take a step back and think about what this is
actually doing. We're computing a representation for every node in our graph,
so let's assume we are currently dealing with a particular node.

We see that we only have a single learnable parameter, $k_0$, and if we ignore
the last normalisation part of the approximation then the first term, $I_N$,
corresponds to the contribution of the node's own features towards its
representation, and the second term corresponds to the contribution from the
node's neighbouring nodes' features.

We see that we're scaling the neighbouring nodes' features by
$\frac{1}{\sqrt{\text{degree}(\textsf{node})}\sqrt{\text{degree}(\textsf{neighbourNode})}}$,
meaning that we are not simply taking the mean of the neighbouring nodes, but
instead we're also considering the *degrees* of the neighbours.

If the neighbour is really "popular", then we do not weigh our connection to it
as that important, but if the neighbour has very few connections and we're one
of those lucky few, then we weigh that connection a lot higher. In a situation
where all nodes have the same degree, this collapses into a simple mean,
however.

So, to sum up, after going through a lot of theoretical hoops we ended up with
the *spectral graph convolution*, which is the graph analogue of the regular
convolution used in CNNs. By approximating this down to a linear stage we end
up with something that is computationally tractable, while still maintaining an
approximation to the spectral graph convolution.


## GCNs in Practice: Implementation

Both [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) and
[Deep Graph Library](https://www.dgl.ai) have implemented GCNs. The code for
the two frameworks is nearly identical. Here is some sample code for PyTorch
Geometric:

```python
import torch
import torch.nn as nn
import torch geometric as tg import torch geometric.nn as tgnn

class GCN(nn.Module):
    def __init__(self, in_feats: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.conv1 = tgnn.GCNConv(in_feats, hidden_size)
        self.conv2 = tgnn.GCNConv(hidden_size, num_classes)

    def forward(self, data: tg.data.Data):
        x, edge index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
```

Lastly, here is some sample code for the Deep Graph Library:

```python
import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn

class GCN(nn.Module):
    def __init__(self, in_feats: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in feats, hidden size)
        self.conv2 = dglnn.GraphConv(hidden size, num classes)

    def forward(self, graph: dgl.DGLGraph, x: torch.tensor):
        x = self.conv1(graph, x)
        x = torch.relu(x)
        x = self.conv2(graph, x)
        return x
```
