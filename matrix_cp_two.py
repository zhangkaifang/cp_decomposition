# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""=====================================
@author : kaifang zhang
@time   : 2021/7/16 3:14 下午
@contact: 1115291605@qq.com
====================================="""
import numpy as np
from numpy.linalg import inv as inv


def kr_prod(a, b):
    return np.einsum('ir, jr -> ijr', a, b).reshape(a.shape[0] * b.shape[0], -1)


def CP_ALS(sparse_mat, m, n, k=5, lam=0.0001, nmax=10000):
    """
    矩阵缺失元素补全使用交替最小二乘法！
    """
    W = np.random.rand(m, k)
    X = np.random.rand(n, k)
    binary_mat = np.zeros((m, n))
    position = np.where((sparse_mat != 0))
    binary_mat[position] = 1
    for iter in range(nmax):
        var1 = X.T
        var2 = kr_prod(var1, var1)
        var3 = np.matmul(var2, binary_mat.T)
        var4 = np.matmul(var1, sparse_mat.T)
        for i in range(m):
            W[i, :] = np.matmul(inv((var3[:, i].reshape([k, k])) + lam * np.eye(k)), var4[:, i])

        var1 = W.T
        var2 = kr_prod(var1, var1)
        var3 = np.matmul(var2, binary_mat)
        var4 = np.matmul(var1, sparse_mat)
        for i in range(n):
            X[i, :] = np.matmul(inv((var3[:, i].reshape([k, k])) + lam * np.eye(k)), var4[:, i])
    return np.dot(W, X.T)


if __name__ == '__main__':
    """ R(n, T)
    评分矩阵R=PQ
    P: 初始化用户特征矩阵n * k
    Q：初始化物品特征矩阵k * T
    """
    R = np.array([[4, 0, 2, 0, 1],
                  [0, 0, 2, 3, 1],
                  [4, 1, 2, 0, 1],
                  [4, 1, 2, 5, 1],
                  [3, 0, 5, 0, 2],
                  [1, 0, 3, 0, 4]])
    mat = CP_ALS(R, R.shape[0], R.shape[1])
    print(mat)
