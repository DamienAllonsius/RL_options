import numpy as np


def sample_cdf(cum_probs):  # cumulative dictribution function
    rand = np.random.rand()
    return sum(cum_probs < rand)


def sample_pmf(probs):  # probability mass function
    probs = np.array(probs)
    assert sum(probs) >= 0.9999999, "this vector does not sum to 1. We need a proper probability mass function"
    return sample_cdf(probs.cumsum())
