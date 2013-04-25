# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt

# 生成するサンプル数
N = 1000

def quad(x,A):
    return np.array(np.matrix(x).T*np.matrix(A)*np.matrix(x))

def gaussian(x, mu, Sigma):
    """多変量ガウス関数"""
    tmp1 = 1 / ((2*np.pi) ** (x.shape[0]/2.0))
    tmp2 = 1 / (np.linalg.det(Sigma) ** 0.5)
    tmp3 = - 0.5 * quad(x-mu,np.linalg.inv(Sigma))
    return tmp1 * tmp2 * np.exp(tmp3)

def sample_gaussian(n, mu, Sigma):
    """多変量ガウス関数のサンプル生成"""
    X = np.dot(np.real(la.sqrtm(Sigma)),np.random.randn(mu.shape[0],n)) + np.tile(mu, [1,n])
    return X

def mle(X):
    mu_mle = np.mean(X,axis=1)      # 列ベクトルの平均をとったのに，戻り値が行ベクトルになってしまう！
    mu_mle = np.array([mu_mle]).T   # 2次元化し，転置して列ベクトルにする
    tmp = (X-np.tile(mu,[1,N]))
    Sigma_mle = np.zeros((X.shape[0],X.shape[0]))
    for n in range(N):
        Sigma_mle += np.array(np.matrix(tmp[:,n]).reshape(X.shape[0],1)*np.matrix(tmp[:,n]).reshape(1,X.shape[0]))/N
    return mu_mle, Sigma_mle

def draw(ax, X, mu, Sigma):
    # 描画のクリア
    ax.collections = []

    # データをプロット
    ax.scatter(X[0,:], X[1,:], c='g', marker='x')

    # 等高線を描画
    xlist = np.linspace(-5, 5, 50)
    ylist = np.linspace(-5, 5, 50)
    x,y = np.meshgrid(xlist,ylist)
    z = np.zeros((50,50))
    for i in range(len(ylist)):
        for j in range(len(xlist)):
            xx = np.array([[xlist[j]], [ylist[i]]])
            z[i,j] = gaussian(xx, mu, Sigma)
    ax.contour(x, y, z, np.linspace(0.01,0.5,10), colors='k', linewidth=1)

    # ガウス分布の平均を描画
    ax.scatter(mu[0], mu[1], c='r', marker='o')

    # 軸の調整
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('2D Normal distribution')
    ax.grid()

    plt.draw()

if __name__ == "__main__":

    # 描画関数用初期化処理
    plt.ion()         # インタラクティブモード
    ax1 = plt.figure().add_subplot(121,aspect='equal')
    ax2 = plt.subplot(122,aspect='equal')

    # 平均、分散を初期化
    mu = np.array([[1],[2]])    # 列ベクトル
    #mu = 6*np.random.rand(2,1)-3
    Sigma = np.array([[3, 1], [2, 1]])
    #A = np.random.randn(2,2)+np.eye(2)
    #Sigma = np.dot(A.T,A)

    # データサンプル生成
    X = sample_gaussian(N, mu, Sigma)

    # 最尤推定
    mu_mle,Sigma_mle = mle(X)

    ## 2次元正規分布の描画
    draw(ax1, X, mu, Sigma)             # 正解
    draw(ax2, X, mu_mle, Sigma_mle)     # 推定結果

