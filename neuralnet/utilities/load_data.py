import numpy as np 


def polynomial_data(num_points, noise, w):
    dim = w.size - 1
    # Generate feature vector 
    x = np.random.normal(size=(num_points, 1))
    x1 = np.power(x, 0)
    for d in range(dim):
        x1 = np.concatenate((np.power(x, 1 + d), x1), axis=1)  # X = [x, 1].
    y = np.dot(x1, w) + np.random.normal(size=(num_points,)) * noise  # y = Xw + eps

    return x1, y


def linear_separable_data(num_positive, num_negative=None, noise=0., offset=1, dim=2):
    if num_negative is None:
        num_negative = num_positive

    x = offset + noise * np.random.randn(num_positive, dim)
    y = 1 * np.ones((num_positive,), dtype=np.int)

    x = np.concatenate((x, noise * np.random.randn(num_negative, dim)), axis=0)
    y = np.concatenate((y, -1 * np.ones((num_negative,), dtype=np.int)), axis=0)

    x = np.concatenate((x, np.ones((num_positive + num_negative, 1))), axis=1)

    return x, y


def circular_separable_data(num_positive, num_negative=None, noise=0., offset=1, dim=2):
    if num_negative is None:
        num_negative = num_positive
    x = np.random.randn(num_positive, dim)
    x = offset * x / np.linalg.norm(x, axis=1, keepdims=True)  # Normalize datapoints to have norm 1.
    x += np.random.randn(num_positive, 2) * noise;
    y = 1 * np.ones((num_positive,), dtype=np.int)

    x = np.concatenate((x, noise * np.random.randn(num_negative, dim)), axis=0)
    y = np.concatenate((y, -1 * np.ones((num_negative,), dtype=np.int)), axis=0)
    x = np.concatenate((x, np.ones((num_positive + num_negative, 1))), axis=1)

    return x, y
