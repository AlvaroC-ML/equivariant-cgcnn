### This project is a work in progress. The project is written in Tensorflow and uses [Spektral](https://graphneural.network/) as a framework for graph neural networks.

# Equivariant Crystal Graph Convolutional Neural Network (CGCNN)

## Background

A crystal can be modeled as a graph where each vertex in the graph is an atom of the crystal. Edge definitions vary from model to model, but a popular definition is one based on distance. That is, two nodes are adjacent if their distance is below a specified cut-off point.

Modeling a crystal as a graph has advantages. The field of graph neural networks (GNNs) is a rapidly evolving field of machine learning models that accept graphs as inputs. GNNs can make node level predictions or graph level predictions. In my project at the National Renewable Energy Laboratory (NREL), we focused on graph level predictions. 

## Objective

The objective is to calculate the anisotropic thermal conductivity of a crystal. If you do not know enough chemistry to understand what that means, don't worry. Me neither. So let's interpret the concept as a black box through mathematics.

Thermal conductivity in some crystals varies according to the direction in which the crystal is moving. Let $\kappa_L(C, u)$ be a measure of the thermal conductivity of the crystal $C$ when $C$ moves in direction $u$ where $u$ is a unitary vector. For any rotation or translation $R$, we have

$$\kappa_L(R(C), R(u))=\kappa_L(C, u).$$

## Optimizing our ML model

By definition, $\kappa_L(C, u)=u^TTu$ where $T$ is a rank 2 tensor. This tensor depends only on $C$. That is, if we had another direction $v$, we would still have $\kappa_L(C, v)=v^TTv$. Thus, to predict $\kappa_L$ for every crystal $C$ and direction $u$, it suffices to predict $T$ for every crystal $C$. Let $M$ be our model, so we want $M(C)=T$. We can mathematically prove that for $M$ to be exact, we need it to be **rotationally equivariant**. That is, we need

$$M(R(C)) = R(M(C)).$$

Plans might change, but to predict $T$ we plan to implement the [PAINN model](https://arxiv.org/pdf/2102.03150.pdf), originally written in PyTorch, in Tensorflow. For a baseline model, we will use a slight variation of a standard Crystal Graph Convolutional Neural Network as described in [this paper](https://www.sciencedirect.com/science/article/pii/S2666389921002233) to calculate the maximum possible values of $\kappa_L$.

The dataset containing values of $T$ is publicly available [here](https://github.com/prashungorai/anisotropy-atlas/blob/master/cm2020-kappaL/kappaL-tensors-layered.csv). 
