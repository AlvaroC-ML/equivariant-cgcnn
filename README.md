### To see the paper explaining the math behind the project, [click here](https://drive.google.com/file/d/163PP7nIyVmASueEns9KbkwJ7s--6VunG/view?usp=sharing).

### For a very brief and graphical description of the project (a poster!), [click here](https://drive.google.com/file/d/1aFhOf2-nh2RjweBDDC97irVNO4ESjT8e/view?usp=sharing).

# Equivariant Crystal Graph Convolutional Neural Network (CGCNN)

## Background

A crystal can be modeled as a graph where each vertex in the graph is an atom of the crystal. Edge definitions vary from model to model, but a popular definition is one based on distance. That is, two nodes are adjacent if their distance is below a specified cut-off point.

Modeling a crystal as a graph has advantages. The field of graph neural networks (GNNs) is a rapidly evolving field of machine learning models that accept graphs as inputs. GNNs can make node level predictions or graph level predictions. In my project at the National Renewable Energy Laboratory (NREL), we focused on graph level predictions. 

## Objective

The objective is to calculate the anisotropic thermal conductivity of a crystal. If you do not know enough chemistry to understand what that means, don't worry. Me neither. So let's interpret the concept as a black box through mathematics.

Thermal conductivity in some crystals varies according to the direction in which the crystal is moving. Let $\kappa_L(C, u)$ be a measure of the thermal conductivity of the crystal $C$ when $C$ moves in direction $u$ where $u$ is a unitary vector. $\kappa_L$ has three properties. For any rotation or translation $R$, 
$$\kappa_L(R(C), R(u))=\kappa_L(C, u),$$
$$\kappa_L(R(C), u)=\kappa_L(C, R^{-1}(u)),$$
and
$$\kappa_L(C, R(C))=\kappa_L(R^{-1}(C), u).$$
How do we make a model that, by design, respects these rules?

## Optimizing our ML model

By definition, $\kappa_L(C, u)=u'Tu$ where $T$ is a rank 2 symmetric tensor. This tensor depends only on $C$. That is, if we had another direction $v$, we would still have $\kappa_L(C, v)=v'Tv$. Thus, to predict $\kappa_L$ for every crystal $C$ and direction $u$, it suffices to predict $T$ for every crystal $C$. Let $M$ be our model, so we want $M(C)=T$. To respect the desired properties, we need a **rotationally equivariant** model. That is, we need

$$M(R(C)) = R(M(C)).$$

But what does it mean to rotate a rank 2 tensor? We can mathematically prove that it is enough to obtain:

$$M(R(C))=RTR'$$

where $R$ is the corresponding rotation matrix. We designed a GNN architecture such that, if $M(C) = T$, then $M(R(C)) = RTR'$, always. This was achieved by using [Geometric Vector Perceptrons](https://openreview.net/pdf?id=1YLJDvSx6J4). For a baseline model, we will use a slight variation of a standard Crystal Graph Convolutional Neural Network as described in [this paper](https://www.sciencedirect.com/science/article/pii/S2666389921002233) to calculate the maximum possible values of $\kappa_L$. Details can be found on the paper at the top.

The dataset containing values of $T$ is publicly available [here](https://github.com/prashungorai/anisotropy-atlas/blob/master/cm2020-kappaL/kappaL-tensors-layered.csv). 
