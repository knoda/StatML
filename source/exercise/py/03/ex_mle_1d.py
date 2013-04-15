# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt

# 生成するサンプル数
N = 10

def gaussian(x, mu, sigma2):
    """1次元ガウス関数"""
    tmp1 = 1 / ((2*np.pi) ** 0.5)
    tmp2 = 1 / (sigma2 ** 0.5)
    tmp3 = - 0.5 * ((x-mu) ** 2) / (sigma2 ** 0.5)
    return tmp1 * tmp2 * np.exp(tmp3)

def sample_gaussian(n, mu, sigma2):
    """1次元ガウス関数のサンプル生成"""
    X = (sigma2 ** 0.5) * np.random.randn(n)+mu
    return X

def mle(X):
    mu_mle = np.mean(X)
    sigma2_mle = np.std(X)
    return mu_mle, sigma2_mle

def draw(ax, X, mu, sigma2, mu_mle, sigma2_mle):
    # 描画のクリア
    ax.collections = []
    ax.lines = []

    # 訓練データを描画
    ax.plot(X[:], np.zeros((X.size,1)), 'gx')

    # ガウス分布の平均を描画
    y = gaussian(mu, mu, sigma2)
    ax.scatter(mu, y, c='r', marker='o')

    # ガウス分布を描画
    xlist = np.linspace(-2.5, 2.5, 50)
    ylist = []
    for n in range(len(xlist)):
        y = gaussian(xlist[n], mu, sigma2)
        ylist.append(y)
    ax.plot(xlist, ylist, 'k-')

    # ガウス分布を描画
    xlist = np.linspace(-2.5, 2.5, 50)
    ylist = []
    for n in range(len(xlist)):
        y = gaussian(xlist[n], mu_mle, sigma2_mle)
        ylist.append(y)
    ax.plot(xlist, ylist, 'r-')

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.1, 0.5)

    plt.draw()

if __name__ == "__main__":

    # 描画関数用初期化処理
    plt.ion()         # インタラクティブモード
    ax = plt.figure().add_subplot(111)

    # 平均、分散を初期化
    mu = 0.5
    sigma2 = 1.0

    # データサンプル生成
    X = sample_gaussian(N, mu, sigma2)

    # 最尤推定
    mu_mle,sigma2_mle = mle(X)

    # 1次元正規分布の描画
    draw(ax, X, mu, sigma2, mu_mle, sigma2_mle)

