import itertools
import numpy as np
import tensorflow as tf

class AFE():
    
    # Compute expert's empirical feature expectation
    def comp_afe(self, N, gamma, F):
        afe = np.zeros(len(F[0]))
        for t in range(0, N):
            # print("List of features length: ", len(F))
            # print("Current index: ", t)
            afe += np.multiply((gamma ** t), np.array(F[t]))
        return afe