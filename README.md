# matvec
Source code for the paper "Exploiting Special Structures in Large Sparse Matrix for Efficient Matrix-Vector Multiplication", published at the 13th Bayesian Inference for Stochastic Processes.

## Experiment 1: Fast matvec procedure for matrix with Kronecker structure

- The matvec procedure is significantly faster than the direct multiplication, but the decomposition takes too long.
- Key question is: How far we can generalize Q? If Q is diagonal or block diagonal like the CMB version, it is solved. 
