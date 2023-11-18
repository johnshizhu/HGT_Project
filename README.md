# Heterogenous Graph Transformer (HGT)

Goals: 
1. Implement a Heterogeneous Graph Transformer Architecture.
2. Optimize model processes through leveraging sparsity and quantization. 
3. Introduce distributed training implementation, with Top-K SGD. 

- This project is done ONLY for educational purposes

## Summary and Write-up
The Heterogeneous Graph Transformer(HGT) architecture was designed to perform inference on heterogeneous graph data (multi-type nodes and edges). This architecture leverages the idea of a "meta relation triplet" that is a numerical abstraction of {SOURCE node type, EDGE type, TARGET node type} in order to parameterize attention and message passing within the model.<br>
For more information on transformer reference: https://arxiv.org/abs/1706.03762 <br>
View the full writeup here: https://docs.google.com/document/d/1d9o0fswgC1evu5sxMwIkeSvmzb31Zmho6LytJepAYbs/edit?usp=sharing

Credit for Hetergeneous Graph Transformer goes to the authors of the original paper, Zinui Hu, Yuxiao Dong, Kuansan Wang, and Yizhou Sun
For original paper and implementation, reference: https://dl.acm.org/doi/fullHtml/10.1145/3366423.3380027

## Overview
- HGT/
  - HGT.py - Implementation of the Heterogeneous Graph Transformer Layer
  - mode.py - Implementation of the Heterogeneous Graph Transformer full model + Classifier
  - RTE.py - Implementation of Relative Temporal Encoding (in progress)
  - hgt_utils.py - Various Functions used for Data preprocessing and training
  - mag_experiment.ipynb - Jupyter Notebook for Training original HGT implementation
  - --------------------------------------------------------------------------------
  - sparse_hgt.py - HGT layer implementation w/ sparsity optimizations
  - sparse_model.py - HGT model implementation w/ sparsity optimizations
  - sparse_mag_experiment.ipynb - sparse HGT experimentts

## Dependencies
- Pytorch
- Pytorch_Geometric

This project is currently a work in progress....
