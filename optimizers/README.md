# Overview of Optimization algorithms

## Particle Swarm Optimization
1. Description
2. PseudoCode
3. Advantages/Disadvantages
4. Notes on Python implementation

###

## Nelder Mead
1. Description
   ![](simplex.png)

   Nelder-Mead [NM] method (also called Downhill Simplex) uses a *simplex*, i.e. the 'most simple volume' in the parameter space with **N+1** corner points **p_j**. In every iteration of the algorithm, the point with the worst (that is, maximum) fitness evaluation gets replaced by a better one. If the simplex gets sufficiently small, the fitness values provide a reasonable approximation of the gradient.

2. Algorithm/Pseudocode
   
   Denote the following definitions:
   * the best point: **F_best = F(p_best)**
   * the worst points: **F_worst = F(p_worst)**, and **F_(worst-1) = F(p_(worst-1))**
   * the barycenter: **p_bary**, centroid of all ***but*** the worst point
   * the search line **L** along two points, which can be defined as **L = p + a(p - q)** for some parameter **a** and points **p** and **q**
   * the parameters:
     * **alpha** > 0 (reflection)
     * **beta**  with beta > alpha (expansion)
     * **gamma** between 0 and 1 (contraction)
     * **sigma** between 0 and 1 (shrinking)


   We start of with an initial simplex, i.e. the **N+1** corner points, and sort them according to the fitness value in order to calculate the candidates above. Then, for every iteration, we do the following:
   ```
   p_cand1 = p_bary + alpha * (p_bary - p_worst)

   if F(p_cand1) < F_best:
      - try p_cand2 = p_bary + beta * (p_bary - p_worst)    # with beta > alpha
      - replace p_worst by better of p_cand1, p_cand2

   else if F(p_cand1) >= F_best && F(p_cand1) < F_(worst-1):
      - replace p_worst by p_cand1

   else if F(p_cand1) < F_worst:
      # p_cand1 is better then the worst, but not as good as worst-1
      # --> new candidate outside of simplex, but not as far as p_cand1

      - try p_cand2' = p_cand1 + gamma * (p_bary - p_cand1) # gamma < 1
      - replace p_worst by p_cand2' if better
        else SHRINK

   else if F(p_cand1) >= F_worst:
      # new candidate inside the simplex
      - try p_cand2' = p_worst + gamma * (p_bary - p_worst)
      - replace p_worst by p_cand2' if better
        else SHRINK

   SHRINK: # nothing was succesful, so just shrink the simplex
      - replace p_j by p_j' = p_j + sigma * (p_best - p_j)
        for every j
   ```

   A common choice of parameters is alpha = 1, beta = 2, gamma = 0.5 = sigma.

3. Advantages/Disadvantages
   
   | Advantages                    | Disadvantages                    |
   | ----------------------------- | -------------------------------- |
   | - TBD                         | TBD                              |

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