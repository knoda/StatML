# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt

# 生成するサンプル数
N1 = 150
N2 = 150
#N1 = 20
#N2 = 180
#N1 = 180
#N2 = 20

def gaussian(x, mu, Sigma):
    """多変量ガウス関数"""
    tmp1 = 1 / ((2*np.pi) ** (x.size/2.0))
    tmp2 = 1 / (np.linalg.det(Sigma) ** 0.5)
    tmp3 = - 0.5 * np.dot(np.dot(x-mu,np.linalg.inv(Sigma)), x-mu)
    return tmp1 * tmp2 * np.exp(tmp3)

def sample_gaussian(n, mu, Sigma):
    """多変量ガウス関数のサンプル生成"""
    X = np.dot(np.real(la.sqrtm(Sigma)),np.random.randn(mu.size,n)) + np.tile(mu, [n,1]).T
    return X

def vecmul(X,mu,NN):
    N = X.shape[1]
    Sigma = np.zeros((X.shape[0],X.shape[0]))
    tmp = (X-np.tile(mu,[N,1]).T)
    for n in range(N):
        Sigma += np.matrix(tmp[:,n]).reshape(X.shape[0],1)*np.matrix(tmp[:,n]).reshape(1,X.shape[0])/NN
    return Sigma

def quad(mu,Sigma):
    return np.array(np.matrix(mu).reshape(1,mu.shape[0])*np.matrix(np.linalg.inv(Sigma))*np.matrix(mu).reshape(mu.shape[0],1))

def lda(X1, X2):
    mu1_lda = np.mean(X1,axis=1)
    mu2_lda = np.mean(X2,axis=1)

    Sigma_lda = np.zeros((X1.shape[0],X1.shape[0]))
    Sigma_lda += vecmul(X1,mu1_lda,N1+N2)
    Sigma_lda += vecmul(X2,mu2_lda,N1+N2)

    a = np.dot(np.linalg.inv(Sigma_lda), mu1_lda-mu2_lda)
    b = -0.5 * (quad(mu1_lda,Sigma_lda) - quad(mu2_lda,Sigma_lda)) + np.log(np.float(N1)/N2)

    return a,b

def draw(ax, X1, X2, a, b):
    # 描画のクリア
    ax.collections = []

    # データをプロット
    ax.scatter(X1[0,:], X1[1,:], c='b', marker='o')
    ax.scatter(X2[0,:], X2[1,:], c='r', marker='x')

    # 境界線の描画
    x = np.linspace(-10,10,50)
    y = np.zeros(50)
    for i in range(len(x)):
        y[i] = (-b-a[0]*x[i])/a[1]
    ax.plot(x,y,'k-')

    # 軸の調整
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('2D Normal distribution')
    ax.grid()

    plt.draw()

if __name__ == "__main__":

    # 描画関数用初期化処理
    plt.ion()         # インタラクティブモード
    ax = plt.figure().add_subplot(111)

    # 平均、分散を初期化
    mu1 = np.array([2,0])
    mu2 = np.array([-2,0])
    Sigma = np.array([[1, 0], [0, 9]])
    #beta = -np.pi/4
    #Sigma = np.array([[9-8*(np.cos(beta)**2), 8*np.sin(beta)*np.cos(beta)], \
    #[8*np.sin(beta)*np.cos(beta), 9-8*(np.sin(beta)**2)]])

    # データサンプル生成
    X1 = sample_gaussian(N1, mu1, Sigma)
    X2 = sample_gaussian(N2, mu2, Sigma)

    # 線形判別
    a,b  = lda(X1,X2)

    # 2次元正規分布の描画
    draw(ax, X1, X2, a, b)
