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

def lda(X1, X2):
    N1 = X1.shape[1]
    N2 = X2.shape[1]

    mu1 = np.mean(X1,axis=1)
    mu1 = np.array([mu1]).T
    mu2 = np.mean(X2,axis=1)
    mu2 = np.array([mu2]).T

    Sigma = np.zeros((X1.shape[0],X1.shape[0]))
    Sigma += vecmul(X1,mu1,N1+N2)
    Sigma += vecmul(X2,mu2,N1+N2)

    return mu1, mu2, Sigma

def draw(t,ind):
    for i in range(ind.size):
        ax = plt.subplot(1,ind.size,i)
        ax.imshow(t[:,ind[i]].reshape(16,16))
    plt.bone()
    plt.show()

if __name__ == "__main__":

    # 手書き数字データの読み込み
    data = io.loadmat('../data/digit.mat')
    X = data['X']
    T = data['T']

    # 線形判別分析
    mu1, mu2, Sigma = lda(X[:,:,1],X[:,:,2])  # 2と3の識別
    invS = np.linalg.inv(Sigma)

    # カテゴリの事後確率の計算 (複数データ)
    t = T[:,:,1]  # 2のデータでテスト
    p1 = np.array(np.matrix(t).T*np.matrix(invS)*np.matrix(mu1) - quad(mu1,invS)/2)
    p2 = np.array(np.matrix(t).T*np.matrix(invS)*np.matrix(mu2) - quad(mu2,invS)/2)

    result =  np.sign(p1-p2)
    ind = np.where(result == -1)

    # 2を3と間違えたデータの表示
    draw(t,ind[0])

