# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as io
import matplotlib as mpl
import matplotlib.pyplot as plt

def gaussian(x, mu, Sigma):
    """多変量ガウス関数"""
    tmp1 = 1 / ((2*np.pi) ** (x.size/2.0))
    tmp2 = 1 / (np.linalg.det(Sigma) ** 0.5)
    tmp3 = - 0.5 * np.dot(np.dot(x-mu,np.linalg.inv(Sigma)), x-mu)
    return tmp1 * tmp2 * np.exp(tmp3)

def vecmul(X,mu,NN):
    N = X.shape[1]
    Sigma = np.zeros((X.shape[0],X.shape[0]))
    tmp = (X-np.tile(mu,[N,1]).T)
    for n in range(N):
        Sigma += np.matrix(tmp[:,n]).reshape(X.shape[0],1)*np.matrix(tmp[:,n]).reshape(1,X.shape[0])/NN
    return Sigma

def quad(mu,Sigma):
    return np.matrix(mu)*np.matrix(Sigma)*np.matrix(mu).T

def lda(X1, X2):
    N1 = X1.shape[1]
    N2 = X2.shape[1]

    mu1 = np.mean(X1,axis=1)
    mu2 = np.mean(X2,axis=1)

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

    # カテゴリの事後確率の計算 (1データ)
    #t = T[:,1,2]
    #p1 = np.array(np.matrix(t)*np.matrix(invS)*np.matrix(mu1).T - quad(mu1,invS)/2)
    #p2 = np.array(np.matrix(t)*np.matrix(invS)*np.matrix(mu2).T - quad(mu2,invS)/2)

    # カテゴリの事後確率の計算 (複数データ)
    t = T[:,:,1]  # 2のデータでテスト
    p1 = np.array(np.matrix(mu1)*np.matrix(invS)*np.matrix(t) - quad(mu1,invS)/2)
    p2 = np.array(np.matrix(mu2)*np.matrix(invS)*np.matrix(t) - quad(mu2,invS)/2)

    result =  np.sign(p1-p2)
    val, ind = np.where(result == -1)

    # 2を3と間違えたデータの表示
    draw(t,ind)

