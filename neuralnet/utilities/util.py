import numpy as np
from .regularizers import Regularizer


def gradient_descent(w0, optimizer, regularizer=None, opts=dict()):
    """
    Mini-Batch Stochastic Gradient descent algorithm.

    w0: is the initial guess
    loss_function: is the loss function you want to optimize. It should have the gradient and loss method.
    opts: a dictionary with the algorithm parameters
    """
    w = w0
    dim = w0.size

    if regularizer is None:
        regularizer = Regularizer()

    eta = opts.get('eta0', 0.01)
    n_iter = opts.get('n_iter', 10)
    batch_size = opts.get('batch_size', 1)
    algorithm = opts.get('algorithm', 'GD')
    n_samples = opts.get('n_samples', optimizer.number_samples)

    indexes = np.arange(0, n_samples, 1)
    if algorithm == 'GD':
        batch_size = n_samples

    trajectory = np.zeros((n_iter + 1, dim))
    trajectory[0, :] = w

    f_val = optimizer.loss(w, indexes)
    f_old = f_val
    grad_sum = 0

    index_traj = np.zeros((n_iter, batch_size), dtype=np.int)

    for it in range(n_iter):
        # Sample indexes.
        # sampling_opts = {'algorithm': opts.get('algorithm', 'GD')}
        # i = sample_indexes(n_samples, batch_size, sampling_opts)
        np.random.shuffle(indexes)
        i = indexes[0:batch_size]
        index_traj[it, :] = i

        # Compute Gradient
        gradient = optimizer.gradient(w, i)
        reg_gradient = regularizer.gradient(w)
        grad_sum += np.sum(np.square(gradient + reg_gradient))

        # Update learning rate.
        learning_rate_opts = {'learning_rate_scheduling': opts.get(
            'learning_rate_scheduling', None),
            'eta0': opts.get('eta0', 0.01),
            'it': it,
            'f_increased': (f_val > f_old),
            'grad_sum': grad_sum}
        eta = compute_learning_rate(eta, learning_rate_opts)

        # Perform gradient step.
        w = w - eta * gradient

        # Regularization
        if opts.get('shrinkage', False):
            wplus = np.abs(w) - eta * regularizer.get_lambda()
            wplus[wplus < 0] = 0
            wplus[-1] = np.abs(w[-1])
            w = np.sign(w) * wplus
        else:
            if batch_size > 0:
                w = w - eta * reg_gradient / batch_size
            else:
                w = w - eta * reg_gradient

        # Compute new cost and save weights.  
        f_old = f_val
        f_val = optimizer.loss(w, indexes)

        trajectory[it + 1, :] = w

    return trajectory, index_traj


def compute_learning_rate(eta, opts=dict()):
    learning_rate_scheduling = opts.get('learning_rate_scheduling', None)
    eta0 = opts.get('eta0', eta)
    f_increased = opts.get('f_increased', False)
    grad_sum = opts.get('grad_sum', 0)
    reg = opts.get('reg', 1/100.)
    it = opts.get('it', 0)
    if learning_rate_scheduling is None:
        eta = eta0  # keep it constant. 
    elif learning_rate_scheduling == 'Annealing':
        eta = eta0 / np.power(it + 1, 0.6)
    elif learning_rate_scheduling == 'Bold driver':
        eta = (eta / 5) if f_increased else (eta * 1.1)
    elif learning_rate_scheduling == 'AdaGrad':
        eta = eta0 / np.sqrt(grad_sum)
    elif learning_rate_scheduling == 'AnnealingSVM':
        eta = min([eta0, 1 / reg * (it + 1.)])
    else:
        raise ValueError('Learning rate scheduling {} not understood'.format(
            learning_rate_scheduling))
    return eta


def dist(X1, X2=None):
    # Build a distance matrix between the elements of X1 and X2.
    if X2 is None:
        X2 = X1

    rows = X1.shape[0]

    if X2.shape[0] == X1.shape[1]:
        cols = 1
    else:
        cols = X2.shape[0]

    D = np.zeros((rows, cols))

    for row in range(rows):
        for col in range(cols):
            if X1.shape[0] == X1.size:
                x1 = X1[row]
            else:
                x1 = X1[row, :]

            if X2.shape[0] == X2.size:
                if cols == 1:
                    x2 = X2
                else:
                    x2 = X2[col]
            else:
                x2 = X2[col, :]

            D[row, col] = np.linalg.norm(x1 - x2)
    return D
