### Implementation of the simulated annealing algorithm 
# Author: Alexander Haegele
# References: 
# - Wikipedia https://en.wikipedia.org/wiki/Simulated_annealing
# - Lilian Besson: https://github.com/Naereen/notebooks/blob/master/Simulated_annealing_in_Python.ipynb
# - Paul Leuchtmann, Slides and Script to the course Optimization Methods for Engineers at ETH Zurich

import numpy as np

def anneal(func, x0, temperature, maxtemp, mintemp, rand_step, acceptance_prob=0,
            bounds=None, maxiter=10000, disp=False, retall=False, callback=None):
    """
    Minimize a function using the simulated annealing algorithm.

    This algorithm only uses function evaluations and the idea of 
    the technical process of annealing, which is closely related to
    the random-walk technique.

    Parameters
    ----------
    func : callable func(x)
        The objective function to be minimized.
    x0 : ndarray
        Initial guess.
    temperature : callable temperature(mintemp, maxtemp, ratio)
        Predefined cooling scheme depending on temperature bounds and ratio
        to the maximum number of iteratons. Should be a monotonically decreasing function.
    maxtemp : number
        Indicates the maximum temperature.
    mintemp : number
        Indicates the minimum temperature
    rand_step : callable rand_step()
        A function that returns a random step, i.e. returns an ndarray with dimension ``dimensions``.
    acceptance_prob : number, optional
        Indicates which acceptance probability to use, either the standard based on difference to
        f_best, or 1 for difference between f_curr and f_new.
    bounds : tuple of array_likes, optional
        The bounds of the function in every dimension. Must be a tuple of size
        two where every array must be of the shape (dimensions,)
    ftol : number, optional
        Absolute error in func(xopt) between iterations that is acceptable for
        convergence.
    maxiter : int, optional
        Maximum number of iterations to perform.
    disp : bool, optional
        Set to True to print intermediate messages.
    retall : bool, optional
        Set to True to return list of solutions at each iteration.
    callback : callable, optional
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.

    Returns
    -------
    xopt : ndarray
        The position with the best associated function value found.
    fopt : float
        Value of function at best position found: ``fopt = func(xopt)``.
    allpos : list
        The history of positions at each iteration.
    allcost : list
        The history of cost values corresponding to the positions
        at each iteration.
    """

    p_curr = x0
    p_best = x0
    f_curr = func(x0)
    f_best = f_curr
    f_0 = f_curr
    pos_history = [p_curr]
    cost_history = [f_curr]

    prob = p_1 if acceptance_prob == 0 else p_2

    for i in range(maxiter):
        ratio = i / float(maxiter)
        T_i = temperature(mintemp, maxtemp, ratio)
        step = rand_step()
        p_new = p_curr + step
        f_new = func(p_new)
        
        if disp:
            print("Step #{:>d}/{:>d} : T = {:>4.3}, f_curr = {:>4.3}, f_new = {:>4.3}, f_best = {:>4.3}".format(i + 1, maxiter, T_i, f_curr, f_new, f_best))
        
        if f_new < f_curr:
            p_curr, f_curr = p_new, f_new

            if f_new < f_best:
                p_best, f_best = p_best, f_new

        elif prob(f_0, f_curr, f_new, f_best, T_i, i) > np.random.uniform():
            p_curr, f_curr = p_new, f_new
        
        if retall:
            pos_history.append(p_curr)
            cost_history.append(f_curr)

        if callback is not None:
            callback(p_curr)
    

    if retall:
        return p_best, f_best, pos_history, cost_history
    else:
        return p_best, f_best


def p_1(f_0, f_curr, f_new, f_best, T_i, i):
    return np.exp(- (abs(f_new - f_best) / T_i))

def p_2(f_0, f_curr, f_new, f_best, T_i, i):
    return np.power((1 + i), - (abs(f_new - f_curr) / f_0))

def temp_linear(mintemp, maxtemp, ratio):
    return mintemp + (1.0 - ratio) * (maxtemp - mintemp)


if __name__ == "__main__":

    f = lambda x : x ** 2.0
    step = lambda: np.random.uniform(-1.0, 1.0)
    anneal(f, 5, temp_linear, 1000, 0, step, acceptance_prob=0, disp=True, retall=True, maxiter=200)