import numpy as np
def sample_cdf(cum_probs): # cumulative dictribution function
    rand = np.random.rand()
    return sum(cum_probs < rand)

def sample_pmf(probs): #probability mass function
    probs = np.array(probs)
    assert sum(probs) >= 0.9999999, "this vector does not sum to 1. We need a proper probability mass function"
    return sample_cdf(probs.cumsum())


# TEST FUNCTION

# expl =  [1/2, 1/6, 1/6, 1/6]
# samp = [0, 0, 0, 0]
# N = 100000
# for k in range(N):
#     samp[sample_pmf(expl)] += 1

# bl = True
# eps = 0.01
# for k in range(len(samp)):
#     bl *= samp[k]/N < expl[k] + eps
#     bl *= samp[k]/N > expl[k] - eps

# print(bl)
