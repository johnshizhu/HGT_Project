# Heterogenous Graph Transformer (HGT)

Goals: 
1. Implement a Heterogeneous Graph Transformer Architecture.
2. Introduce distributed training implementation, accelerated by forced sparsity to speed up inter-device communication. 

- This project is done ONLY for educational purposes

## Summary and Write-up
Previously, most Graph Neural Networks (GNNs) were designed around homogenous graph structures, where edges and nodes are of all one type. This however is insufficient to represent graphs that have varying node and edge types. HGT is designed the take those essential traits into consideration using the Transformer approach.<br>
For more information on transformer reference: https://arxiv.org/abs/1706.03762 <br>
View the full writeup here: https://docs.google.com/document/d/1d9o0fswgC1evu5sxMwIkeSvmzb31Zmho6LytJepAYbs/edit?usp=sharing

Credit for Hetergeneous Graph Transformer goes to the authors of the original paper, Zinui Hu, Yuxiao Dong, Kuansan Wang, and Yizhou Sun
For original paper and implementation, reference: https://dl.acm.org/doi/fullHtml/10.1145/3366423.3380027

## Overview
- HGT.py - the "body" of implementation of HGT
- hgsampling.py - in progress
- local_access.py - Accessing locally stored data for training/analysis
- ogb_load.py - Accessing online dataset download for training/analysis

## Dependencies
- Pytorch
- Pytorch_Geometric

This project is currently a work in progress....