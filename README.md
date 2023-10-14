# Heterogenous Graph Transformer (HGT)

Goals: 
1. Implement a Heterogeneous Graph Transformer Architecture.
2. Introduce distributed training implementation, accelerated by forced sparsity to speed up inter-device communication. 

## Summary and write-up
Previously, most Graph Neural Networks (GNNs) were designed around homogenous graph structures, where edges and nodes are of all one type. This however is insufficient to represent graphs that have varying node and edge types. 
View the full writeup here: https://docs.google.com/document/d/1d9o0fswgC1evu5sxMwIkeSvmzb31Zmho6LytJepAYbs/edit?usp=sharing

For original paper and implementation, reference: https://dl.acm.org/doi/fullHtml/10.1145/3366423.3380027

## Overview
