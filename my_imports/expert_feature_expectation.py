import itertools
import numpy as np
import tensorflow as tf

class EFE():
    
    # Compute expert's empirical feature expectation
    def comp_efe(self, M, N, gamma, trajectories):
        sum1 = np.zeros(len(trajectories[0][0]))
        sum2 = np.zeros(len(trajectories[0][0]))
        for m in range(0, M):
            F = trajectories[m]
            for t in range(0, N):
                features = [float(x) for x in F[t]]
                sum1 += np.multiply((gamma ** t), np.array(features))
            sum2 += sum1
        efe = sum2 / M
        return efe