# Overview of Optimization algorithms

## Particle Swarm Optimization
1. Description
2. PseudoCode
3. Advantages/Disadvantages
4. Notes on Python implementation

###

## Nelder Mead
1. Description
   
   Nelder-Mead [NM] method (also called Downhill Simplex) uses a *simplex*, i.e. the 'most simple volume' in the parameter space with $N+1$ corner points $p_j$. In every iteration of the algorithm, the point with the worst (that is, maximum) fitness evaluation gets replaced by a better one. If the simplex gets sufficiently small, the fitness values provide a reasonable approximation of the gradient.

2. Algorithm/Pseudocode
   
   Denote the following definitions:
   * the best point: $F_{best} = F(p_{best})$
   * the worst points: $F_{worst} = F(p_{worst})$, $F_{worst-1} = F(p_{worst-1})$
   * the barycenter: $p_{bary}$

   We start of with an initial simplex, i.e. the $N+1$ corner points, and sort them according to the fitness value in order to specify the candidates above. Then, for every iteration, we do the following:

   TBD
3. Advantages/Disadvantages
4. Notes on Python implementation
```
if adaptive:
    rho = 1
    chi = 1 + 2/N
    psi = 0.75 - 1/(2*N)
    sigma = 1 - 1/N
else:
    rho = 1
    chi = 2
    psi = 0.5
    sigma = 0.5

nonzdelt = 0.05
zdelt = 0.00025
```

## Simulated Annealing
   1. Description
   2. PseudoCode
   3. Advantages/Disadvantages
   4. Notes on Python implementation