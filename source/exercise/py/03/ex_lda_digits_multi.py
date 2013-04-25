# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as io
import matplotlib as mpl
import matplotlib.pyplot as plt

def quad(x,A):
    return np.array(np.matrix(x).T*np.matrix(A)*np.matrix(x))

def vecmul(X,mu,NN):
    N = X.shape[1]
    Sigma = np.zeros((X.shape[0],X.shape[0]))
    tmp = (X-np.tile(mu,[1,N]))
    for n in range(N):
        Sigma += np.matrix(tmp[:,n]).reshape(X.shape[0],1)*np.matrix(tmp[:,n]).reshape(1,X.shape[0])/NN
    return Sigma

def lda_multi(X):
    N = 0
    for c in range(X.shape[2]):
        N += X[:,:,c].shape[1]
    mu = np.matrix(np.zeros((X.shape[0],X.shape[2])))
    Sigma = np.zeros((X.shape[0],X.shape[0]))
    for c in range(X.shape[2]):
        mu_tmp = np.mean(X[:,:,c],axis=1)
        mu[:,c] = np.array([mu_tmp]).T
        Sigma += vecmul(X[:,:,c],mu[:,c],N)
    return np.array(mu), Sigma

if __name__ == "__main__":

    # 手書き数字データの読み込み
    data = io.loadmat('../data/digit.mat')
    X = data['X']
    T = data['T']

    # 線形判別分析
    mu, Sigma = lda_multi(X)
    invS = np.linalg.inv(Sigma)

    # カテゴリの事後確率の計算 (複数データ)
    p = np.zeros((T.shape[2],T.shape[1],mu.shape[1]))
    for ct in range(T.shape[2]):
        t = T[:,:,ct]  # ct番目のカテゴリのデータでテスト
        for c in range(mu.shape[1]):
            p[ct,:,c] = (np.matrix(t).T*np.matrix(invS)*np.matrix(mu)[:,c] - quad(np.matrix(mu)[:,c],invS)/2).T

    # 最大事後確率を取るインデックスを抽出
    ind = np.argmax(p,axis=2)

    C = np.zeros((T.shape[2],mu.shape[1]))
    for ct in range(T.shape[2]):
        for c in range(mu.shape[1]):
            C[ct,c] = np.sum(ind[ct,:]==c)

    print C

