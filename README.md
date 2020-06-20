# **Optimization Methods for Engineers** - Paricle Swarm Optimization, Nelder-Mead, Simulated Annealing
This repository includes sample applications and use cases of optimization methods, mainly particle swarm optimization. It was created as part of the course 'Optimization Methods for Engineers' at ETH Zürich.

#### Part 1: Analysing PSO on typical test functions for optimization
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

#### Part 2: Analysing PSO on a neural network


Option 1:
* General discussion of NM, SA, PSO for functions in the context of optimization methods 
  * strengths & weaknesses, comparison
* Analyze different optimizers for our implementation of a neural net (Nelder Mead, Simulated Annealing, PSO)
  * General usability, performance, convergence rate
  * Try to see strengths & weaknesses for this specific application

Option 2:
* The idea is as follows: We select some particular functions that are interesting in the context of optimization methods. For each of those functions, we create a notebook that includes a description of the function, the discussion of particle swarm optimization when applied to this problem as well as the results. Visualizations are always very much appreciated. See the example notebooks as reference on how to implement these ideas.

The current selection of functions is:
  * Rosenbrock https://en.wikipedia.org/wiki/Rosenbrock_function
  * Himmelblau https://en.wikipedia.org/wiki/Himmelblau%27s_function
  * ...

What holds for both: find good heuristics (e.g. good selection of hyperparameters) for optimizers.



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

