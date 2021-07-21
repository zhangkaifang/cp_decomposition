# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""=====================================
@author : kaifang zhang
@time   : 2021/7/13 3:50 下午
@contact: 1115291605@qq.com
====================================="""
import numpy as np
from scipy.linalg import khatri_rao


def ten2mat(tensor, mode):
    """Return mu-mode matricization from a given tensor"""
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')


def mat2ten(mat1, mat2, mat3):
    """Return tensor based on matrix"""
    return np.einsum('ir, jr, tr -> ijt', mat1, mat2, mat3)


def alter_optimization(X, r, nmax=50000):
    n1, n2, n3 = X.shape
    A = np.random.normal(0, 1, (n1, r))  # shape: (4, 10)
    B = np.random.normal(0, 1, (n2, r))  # shape: (7, 10)
    C = np.random.normal(0, 1, (n3, r))  # shape: (22, 10)

    pos = np.where(X != 0)  # where内只有一个参数时，那个参数表示条件，当条件成立时，where返回的是每个符合condition条件元素的坐标,返回的是以元组的形式
    bin_X = np.zeros((n1, n2, n3))
    bin_X[pos] = 1
    X_hat = np.zeros((n1, n2, n3))

    for iters in range(nmax):
        ######################################## optimize A
        var1 = khatri_rao(C, B).T
        var2 = khatri_rao(var1, var1)
        var3 = np.matmul(var2, ten2mat(bin_X, 0).T).reshape([r, r, n1])
        var4 = np.matmul(var1, ten2mat(X, 0).T)
        for i in range(n1):
            var_Lambda = var3[:, :, i]
            inv_var_Lambda = np.linalg.inv((var_Lambda + var_Lambda.T) / 2)
            A[i, :] = np.matmul(inv_var_Lambda, var4[:, i])
        ######################################## optimize B
        var1 = khatri_rao(C, A).T
        var2 = khatri_rao(var1, var1)
        var3 = np.matmul(var2, ten2mat(bin_X, 1).T).reshape([r, r, n2])
        var4 = np.matmul(var1, ten2mat(X, 1).T)
        for j in range(n2):
            var_Lambda = var3[:, :, j]
            inv_var_Lambda = np.linalg.inv((var_Lambda + var_Lambda.T) / 2)
            B[j, :] = np.matmul(inv_var_Lambda, var4[:, j])
        ######################################## optimize C
        var1 = khatri_rao(B, A).T
        var2 = khatri_rao(var1, var1)
        var3 = np.matmul(var2, ten2mat(bin_X, 2).T).reshape([r, r, n3])
        var4 = np.matmul(var1, ten2mat(X, 2).T)
        for t in range(n3):
            var_Lambda = var3[:, :, t]
            inv_var_Lambda = np.linalg.inv((var_Lambda + var_Lambda.T) / 2)
            C[t, :] = np.matmul(inv_var_Lambda, var4[:, t])
        ######################################## Reconstruct tensor
        X_hat = mat2ten(A, B, C)
        loss = np.sum(np.square(X[pos] - X_hat[pos])) / X[pos].shape[0]

        if (iters + 1) % 100 == 0:
            print('迭代次数:', iters, '代价函数:', loss)

    return X_hat


if __name__ == '__main__':
    tensor = np.load('./data/tensor.npy')
    X_hat = alter_optimization(tensor, 10)
    print(tensor.shape)

