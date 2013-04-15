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
    Sigma = np.std(X, axis=0)
    # 列ごとにデータを正規化
    for i in range(col):
        X[:,i] = (X[:,i] - mu[i]) / Sigma[i]
    return X

def gaussian(x, mu, Sigma):
    """多変量ガウス関数"""
    tmp1 = 1 / ((2*np.pi) ** (x.size/2.0))
    tmp2 = 1 / (np.linalg.det(Sigma) ** 0.5)
    tmp3 = - 0.5 * np.dot(np.dot(x-mu, np.linalg.inv(Sigma)), x-mu)
    return tmp1 * tmp2 * np.exp(tmp3)

def likelihood(X, mu, Sigma, w):
    """対数尤度関数"""
    sum = 0.0
    for n in range(len(X)):
        tmp = 0.0
        for k in range(K):
            tmp += w[k] * gaussian(X[n], mu[k], Sigma[k])
        sum += np.log(tmp)
    return sum

def estep(X, mu, Sigma, w):
    for n in range(N):
        # 分母はkによらないので最初に1回だけ計算
        denom = 0.0
        for j in range(K):
            #print cnt, Sigma[j], np.linalg.det(Sigma[j])
            denom += w[j] * gaussian(X[n], mu[j], Sigma[j])

        # 各kについて媒介変数を計算
        for k in range(K):
            gamma[n][k] = w[k] * gaussian(X[n], mu[k], Sigma[k]) / denom
    return gamma

def mstep(X, mu, Sigma, w, gamma):
    for k in range(K):
        # 媒介変数のsumを計算
        Nk = 0.0
        for n in range(N):
            Nk += gamma[n][k]

        # 平均を再計算
        mu[k] = np.array([0.0, 0.0])
        for n in range(N):
            mu[k] += gamma[n][k] * X[n]
        mu[k] /= Nk

        # 共分散を再計算
        Sigma[k] = np.array([[0.0,0.0], [0.0,0.0]])
        for n in range(N):
            tmp = X[n] - mu[k]
            Sigma[k] += gamma[n][k] * np.matrix(tmp).reshape(2,1) * np.matrix(tmp).reshape(1,2)   # 縦ベクトルx横ベクトル
        Sigma[k] /= Nk

        # 混合係数を再計算
        w[k] = Nk / N
    return mu, Sigma, w

def draw(ax, mu, Sigma):
    # 描画のクリア
    ax.collections = []

    # 訓練データを描画
    ax.plot(X[:,0], X[:,1], 'gx')

    # ガウス分布の平均を描画
    for k in range(K):
        ax.scatter(mu[k, 0], mu[k, 1], c='r', marker='o')

    # 等高線を描画
    xlist = np.linspace(-2.5, 2.5, 50)
    ylist = np.linspace(-2.5, 2.5, 50)
    x, y = np.meshgrid(xlist, ylist)
    for k in range(K):
        z = mpl.mlab.bivariate_normal(x, y, np.sqrt(Sigma[k,0,0]), np.sqrt(Sigma[k,1,1]), mu[k,0], mu[k,1], Sigma[k,0,1])
        cs = ax.contour(x, y, z, 3, colors='k', linewidth=1)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)

    plt.draw()

if __name__ == "__main__":

    # 描画関数用初期化処理
    plt.ion()         # インタラクティブモード
    ax = plt.figure().add_subplot(1,1,1)

    # 訓練データをロード
    data = np.loadtxt("../data/faithful.txt")
    X = data[:, 0:2]
    X = scale(X)      # データを正規化（各次元が平均0、分散1になるようにする）
    N = len(X)        # データ数

    # 訓練データから混合ガウス分布のパラメータをEMアルゴリズムで推定する

    # 平均、分散、混合係数を初期化
    mu = np.random.rand(K, 2)
    Sigma = np.zeros((K, 2, 2))
    for k in range(K):
        Sigma[k] = [[1.0, 0.0], [0.0, 1.0]]
    w = np.random.rand(K)

    # 媒介変数の空配列を用意
    gamma = np.zeros((N,K))

    # 対数尤度の初期値を計算
    lh = likelihood(X, mu, Sigma, w)

    cnt = 0
    while 1:
        print cnt, lh

        # E-step: 現在のパラメータを使って、媒介変数を計算
        gammma = estep(X, mu, Sigma, w)

        # M-step: 現在の媒介変数を使って、パラメータを更新
        mu, Sigma, w = mstep(X, mu, Sigma, w, gamma)

        # 推定結果の描画
        draw(ax, mu, Sigma)

        # 収束判定
        lh_new = likelihood(X, mu, Sigma, w)
        diff = lh_new - lh
        if diff < 0.001:
            break
        lh = lh_new
        cnt += 1

