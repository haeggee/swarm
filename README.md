# **Optimization Methods for Engineers** - Paricle Swarm Optimization, Nelder-Mead, Simulated Annealing
This repository includes sample applications and use cases of optimization methods, mainly particle swarm optimization. It was created as part of the course 'Optimization Methods for Engineers' at ETH Zürich.

#### Part 1: Analysing optimizers on typical test functions for optimization
##### **Test function 1 (Rastrigin?):**
1. Optimization with PSO:
    * Optimization
    * Plot cost history
    * Visualization in 2D
    * Visualization in 3D 
    * Analysis of good choice of hyperparameters 
    
    
2. Comparison with Nelder Mead:
    * Performance
    * Convergence rate
    * Complexity
    
    
3. Comparison with Simulated Annealing: 
    * Performance
    * Convergence rate
    * Complexity


##### **Test function 2 (TBD):**
1. Optimization with PSO:
    * Optimization
    * Plot cost history
    * Visualization in 2D
    * Visualization in 3D 
    * Analysis of good choice of hyperparameters 
    
    
2. Comparison with Nelder Mead:
    * Performance
    * Convergence rate 
    * Complexity
    
    
3. Comparison with Simulated Annealing: 
    * Performance
    * Convergence rate
    * Complexity

##### **Discussion:**
1. Strengths
2. Weaknesses

#### Part 2: Analysing PSO, NM and SA for neural network training
Neural networks are a way of parametrizing non-linear functions. Usually, the training of a neural network is done via different variants of gradient descent, i.e. trying to find a good choice of weights by minimizing the loss function.

What we do here in this notebook is another approach: instead of using a gradient based method that does the fitting of the network, we want to apply the particle swarm optimization, simulated annealing and Nelder Mead algorithms to find a good choice of weights. Our application is classical binary classification. The discussion includes:
* Our own implementation of a neural net, forward propagation and loss function
* An interactive example and visualization for binary classification, where one can choose different hyperparameters
* Usability and performance of these optimizers for neural net training


## Developing

All notebooks depend on and use Matplotlib, NumPy, PySwarms, Sklearn, ... (to be extended)

Please consider the rule to always push your changes to the notebook only **AFTER** restarting the kernel and executing all cells starting from the first to ensure a working notebook.

## Bonus
* **Neural Network:** ~~If time is enough, a nice idea is to implement and fine-tune a certain neural-network for an existing dataset (see https://pyswarms.readthedocs.io/en/latest/examples/usecases/train_neural_network.html as a reference)~~ Implemented! TODO-list can be found in the notebook.
* **Bee algorithm:** Also, it would be cool the test the bees algorithm (https://en.wikipedia.org/wiki/Bees_algorithm), which is a special case of the generic particle swarm idea. To implement this, we would have to write our own optimization loop where we differentiate between the different types of bees when updating the positions (scouts, etc.). To see how to implement our own optimization method, we can refer to https://pyswarms.readthedocs.io/en/latest/examples/tutorials/custom_optimization_loop.html

# Credits
Credits belong to:
* The contributors to the PySwarm Toolkit: https://github.com/ljvmiranda921/pyswarms
* Code from Sebastian Curi and Andreas Krause, based on Jaques Grobler (sklearn demos). This was taken from the Introduction to Machine Learning lecture at ETH.

# License
All code in this repository is published under the terms of the [MIT License](LICENSE)

© [Alexander Haegele](https://github.com/haeggee), [Richard von der Horst](https://github.com/RichardVDH), [Paul Elvinger](https://github.com/elvingerpaul)
