A PyToch implementation of the DIM predictive coding model
----------------------------------------------------------

This repositoy contains my replication of:

Nour Eddine, Samer, Trevor Brothers, Lin Wang, Michael Spratling, and Gina R. Kuperberg.
"A Predictive Coding Model of the N400". bioRxiv, 11 April 2023.
https://doi.org/10.1101/2023.04.10.536279.

The code has been tested against Samer's original implementation and approximates it very closely.
In addition, there is a new PyTorch GPU implementation of the model in the form of a stack of PyTorch Modules.

Installation
------------
This package depends on `numpy`, `torch` and `matplotlib`. You probably want to install
these yourself to ensure speedy versions of these compatible with your system.

Next, clone this repository somewhere and install with: `pip install -e .`.

Running
-------
* To reproduce Figures 5 and 6A of the original paper: `examples/n400_simulation/run_simulation.py`
* To run a reference implementation of the N400 model: `examples/n400_simulation_numpy/run_simulation.py`
* To classify the MNIST dataset: `examples/mnist/mnist.py`
