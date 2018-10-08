import itertools
import numpy as np
import tensorflow as tf

class OPT():

    def optimize(self, proj_afe, efe, afe, i):
        # Implementation of the alternate step 2(projection method) of the Abbeel and Ng, 2004 algorithm for apprenticeship learing.
        weights = []
        t = 0.0
        
        if (i > 1):
            diff1 = afe[i - 1] - proj_afe[i - 2]
            diff2 = efe - proj_afe[i - 1]
            proj_afe[i - 1] = proj_afe[i - 2] + (np.dot(diff1,diff2) / np.dot(diff1,diff1)) * diff1
        else:
            proj_afe[0] = afe[0]

        weights = efe - proj_afe[i - 1]
        t = np.linalg.norm(weights)

        return (weights, t)