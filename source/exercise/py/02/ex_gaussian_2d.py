# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def gaussian(x, mu, Sigma):
    """多変量ガウス関数"""
    tmp1 = 1 / ((2*np.pi) ** (x.size/2.0))
    tmp2 = 1 / (np.linalg.det(Sigma) ** 0.5)
    tmp3 = - 0.5 * np.dot(np.dot(x-mu, np.linalg.inv(Sigma)), x-mu)
    return tmp1 * tmp2 * np.exp(tmp3)

def draw(ax, mu, Sigma):
    # 描画のクリア
    ax.collections = []

    # 等高線を描画
    xlist = np.linspace(-5, 5, 50)
    ylist = np.linspace(-5, 5, 50)
    x,y = np.meshgrid(xlist,ylist)
    z = np.zeros((50,50))
    for i in range(len(ylist)):
        for j in range(len(xlist)):
            xx = np.array([xlist[j], ylist[i]])
            z[i,j] = gaussian(xx, mu, Sigma)
    cs = ax.pcolor(x, y, z)
    plt.colorbar(cs)
    plt.jet()
    #plt.bone()

    ax.contour(x, y, z, np.linspace(0.0001,0.5,25), colors='k', linewidth=1)

    # ガウス分布の平均を描画
    ax.scatter(mu[0], mu[1], c='b', marker='x')

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
    ax = plt.figure().add_subplot(1,1,1)

    # 平均、分散を初期化
    #mu = np.array([0, 1])
    mu = 6*np.random.rand(2)-3

    #Sigma = np.array([[5, 0], [0, 1]]);
    Sigma = np.array([[5, 1], [1, 0.5]]);
    #A = np.random.randn(2,2)+np.eye(2)
    #Sigma = np.dot(A.T,A)

    # 2次元正規分布の描画
    draw(ax, mu, Sigma)

