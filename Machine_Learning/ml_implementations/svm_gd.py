# import numpy as np
#
#
# class SVM:
#     def __init__(self, kernel, C=1):
#         self.kernel = kernel
#         self.C = C
#
#     def fit(self, x, y, lr=1e-5, n_iters=400):
#         self.xtrain = x
#         self.ytrain = y
#         self.n = x.shape[0]
#         self.alphas = np.random.random(self.n)
#         self.b = 0
#
#