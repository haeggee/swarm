# Particle Swarm Optimization
This repository includes sample applications and use cases of particle swarm optimizations. It was created as part of the course 'Optimization Methods for Engineers' at ETH Zürich.

The idea is as follows: We select some particular functions that are interesting in the context of optimization methods. For each of those functions, we create a notebook that includes a description of the function, the discussion of particle swarm optimization when applied to this problem as well as the results. Visualizations are always very much appreciated. See the example notebooks as reference on how to implement these ideas.

The current selection of functions is:
* Rosenbrock https://en.wikipedia.org/wiki/Rosenbrock_function
* ...

## Developing

All notebooks depend on and use Matplotlib, NumPy, PySwarms, Sklearn, ... (to be extended)

Please consider the rule to always push your changes to the notebook only **AFTER** restarting the kernel and executing all cells starting from the first to ensure a working notebook.

## Bonus
* **Neural Network:** ~~If time is enough, a nice idea is to implement and fine-tune a certain neural-network for an existing dataset (see https://pyswarms.readthedocs.io/en/latest/examples/usecases/train_neural_network.html as a reference)~~ Implemented! TODO-list can be found in the notebook.
* **Bee algorithm:** Also, it would be cool the test the bees algorithm (https://en.wikipedia.org/wiki/Bees_algorithm), which is a special case of the generic particle swarm idea. To implement this, we would have to write our own optimization loop where we differentiate between the different types of bees when updating the positions (scouts, etc.). To see how to implement our own optimization method, we can refer to https://pyswarms.readthedocs.io/en/latest/examples/tutorials/custom_optimization_loop.html

# Credits
Credits belong to the contributors to the PySwarm Toolkit: https://github.com/ljvmiranda921/pyswarms
