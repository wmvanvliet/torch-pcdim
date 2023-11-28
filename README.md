A Predictive Coding Model of the N400 
-------------------------------------

This repositoy contains my replication of:

Nour Eddine, Samer, Trevor Brothers, Lin Wang, Michael Spratling, and Gina R. Kuperberg.
"A Predictive Coding Model of the N400". bioRxiv, 11 April 2023.
https://doi.org/10.1101/2023.04.10.536279.


The code has been tested against Samer's original implementation.
The CPU version (`model.py`) reproduces the values exactly.

In addition, there is a new PyTorch GPU implementation of the model in the form of a stack of PyTorch Modules.
This does not reproduce the original values exactly, as it skips some of the particularities of the original code in favor of consistancy.

Installation and Running
------------------------

For the CPU model, only `numpy` is required. For the GPU model, you will also need [`torch`](https://pytorch.org/).

Running the CPU model: `simulation.py`  
Running the GPU model: `torch_simulation.py`  

This will reproduce Figure 5 of the original paper.
