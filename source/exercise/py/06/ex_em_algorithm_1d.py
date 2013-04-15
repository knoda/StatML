# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

K = 2   # ガウスモデルの混合数

def scale(X):
    """データ行列Xを列ごとに正規化したデータを返す"""
    # 列の数
    col = X.shape[1]
    # 列ごとに平均値と標準偏差を計算
    mu = np.mean(X, axis=0)
    sigma2 = np.std(X, axis=0)
    # 列ごとにデータを正規化
    for i in range(col):
        X[:,i] = (X[:,i] - mu[i]) / sigma2[i]
    return X

def gaussian(x, mu, sigma2):
    """1次元ガウス関数"""
    tmp1 = 1 / ((2*np.pi) ** 0.5)
    tmp2 = 1 / (sigma2 ** 0.5)
    tmp3 = - 0.5 * ((x-mu) ** 2) / (sigma2 ** 0.5)
    return tmp1 * tmp2 * np.exp(tmp3)

def likelihood(X, mu, sigma2, w):
    """対数尤度関数"""
    sum = 0.0
    for n in range(len(X)):
        tmp = 0.0
        for k in range(K):
            tmp += w[k] * gaussian(X[n], mu[k], sigma2[k])
        sum += np.log(tmp)
    return sum

def estep(X, mu, sigma2, w):
    for n in range(N):
        # 分母はkによらないので最初に1回だけ計算
        denom = 0.0
        for j in range(K):
            #print cnt, sigma2[j], np.linalg.det(sigma2[j])
            denom += w[j] * gaussian(X[n], mu[j], sigma2[j])

        # 各kについて媒介変数を計算
        for k in range(K):
            gamma[n][k] = w[k] * gaussian(X[n], mu[k], sigma2[k]) / denom
    return gamma

def mstep(X, mu, sigma2, w, gamma):
    for k in range(K):
        # 媒介変数のsumを計算
        Nk = 0.0
        for n in range(N):
            Nk += gamma[n][k]

        # 平均を再計算
        mu[k] = 0.0
        for n in range(N):
            mu[k] += gamma[n][k] * X[n]
        mu[k] /= Nk

        # 共分散を再計算
        sigma2[k] = 0.0
        for n in range(N):
            sigma2[k] += gamma[n][k] * ((X[n]-mu[k]) ** 2)
        sigma2[k] /= Nk

        # 混合係数を再計算
        w[k] = Nk / N
    return mu, sigma2, w

def draw(ax, X, mu, sigma2, w):
    # 描画のクリア
    ax.collections = []
    ax.lines = []

    # 訓練データを描画
    ax.plot(X[:], np.zeros((X.size,1)), 'gx')

    # ガウス分布の平均を描画
    for k in range(K):
        y = 0
        for kk in range(K):
            y += w[kk] * gaussian(mu[k], mu[kk], sigma2[kk])
        ax.scatter(mu[k], y, c='r', marker='o')

    # ガウス分布を描画
    xlist = np.linspace(-2.5, 2.5, 50)
    ylist = []
    for n in range(len(xlist)):
        y = 0.0
        for k in range(K):
            y += w[k] * gaussian(xlist[n], mu[k], sigma2[k])
        ylist.append(y)
    ax.plot(xlist, ylist, 'k-')

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.1, 1.1)

    plt.draw()

if __name__ == "__main__":

    # 描画関数用初期化処理
    plt.ion()         # インタラクティブモード
    ax = plt.figure().add_subplot(1,1,1)

    # 訓練データをロード
    data = np.loadtxt("../data/faithful.txt")
    X = data[:, 0:1]
    X = scale(X)      # データを正規化（各次元が平均0、分散1になるようにする）
    N = len(X)        # データ数

    # 訓練データから混合ガウス分布のパラメータをEMアルゴリズムで推定する

    # 平均、分散、混合係数を初期化
    mu = np.random.rand(K)
    sigma2 = np.ones(K)
    w = np.random.rand(K)

    # 媒介変数の空配列を用意
    gamma = np.zeros((N,K))

    # 対数尤度の初期値を計算
    lh = likelihood(X, mu, sigma2, w)

    cnt = 0
    while 1:
        print cnt, lh

        # E-step: 現在のパラメータを使って、媒介変数を計算
        gammma = estep(X, mu, sigma2, w)

        # M-step: 現在の媒介変数を使って、パラメータを更新
        mu, sigma2, w = mstep(X, mu, sigma2, w, gamma)

        # 推定結果の描画
        draw(ax, X, mu, sigma2, w)

        # 収束判定
        lh_new = likelihood(X, mu, sigma2, w)
        diff = lh_new - lh
        if diff < 0.0001:
            break
        lh = lh_new
        cnt += 1

