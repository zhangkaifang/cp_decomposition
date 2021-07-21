#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=====================================
@Author :Kaifang Zhang
@Time   :2021/7/5 1:31
@Contact: kaifang.zkf@dtwave-inc.com
========================================'''
import numpy as np


def LFM_grad_desc(R, K, max_iter, alpha=1e-4, lamda=1e-4):
    """
    实现矩阵缺失元素补全！
    """
    # 基本维度参数定义
    M = len(R)
    N = len(R[0])

    # P、Q初始值，随机生成
    P = np.random.rand(M, K)
    Q = np.random.rand(N, K)
    Q = Q.T
    # 开始迭代
    for step in range(max_iter):
        # 对所有的用户u、物品i做遍历，对应的特征向量Pu，Qi梯度下降
        for u in range(M):
            for i in range(N):
                # 对于每一个大于0的评分，求出预测的评分误差
                if R[u][i] > 0:
                    eui = np.dot(P[u, :], Q[:, i]) - R[u][i]

                    # 带入公式，按照梯度下降算法更新当前的Pu与Qi
                    for k in range(K):
                        P[u][k] = P[u][k] - alpha * (2 * eui * Q[k][i] + 2 * lamda * P[u][k])
                        Q[k][i] = Q[k][i] - alpha * (2 * eui * P[u][k] + 2 * lamda * Q[k][i])

        # u、i遍历完成，所有的特征向量更新完成，可以得到P、Q，可以计算预测评分矩阵
        predR = np.dot(P, Q)

        # 计算当前损失函数
        cost = 0
        for u in range(M):
            for i in range(N):
                if R[u][i] > 0:
                    cost += (np.dot(P[u, :], Q[:, i]) - R[u][i]) ** 2
                    # 加上正则化项
                    for k in range(K):
                        cost += lamda * (P[u][k] ** 2 + Q[k][i] ** 2)
        if step % 1000 == 0:
            print("迭代次数：", step, "损失函数：", cost)
        if cost < 0.001:
            break

    return P, Q.T, cost


if __name__ == '__main__':
    '''
    @输入参数
    R:M*N的评分矩阵
    K:隐特征向量维度
    max_iter:最大迭代次数
    alpha:步长
    lamda:正则化系数
    @输出
    分解之后的P、Q
    P:初始化用户特征矩阵M*k
    Q：初始化物品特征矩阵N*K
    '''
    # 评分矩阵R
    R = np.array([[4, 0, 2, 0, 1],
                  [0, 0, 2, 3, 1],
                  [4, 1, 2, 0, 1],
                  [4, 1, 2, 5, 1],
                  [3, 0, 5, 0, 2],
                  [1, 0, 3, 0, 4]])

    # 给定超参数
    K = 5
    max_iter = 100000
    alpha = 1e-4
    lamda = 1e-3
    P, Q, cost = LFM_grad_desc(R, K, max_iter, alpha, lamda)
    predR = P.dot(Q.T)
    # 预测矩阵
    print(predR)
