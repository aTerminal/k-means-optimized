from os import sep
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np
import csv
from sklearn import decomposition
import time


def normalization(data: np.ndarray) -> np.ndarray:
    max_value = np.max(data, axis=0)
    min_value = np.min(data, axis=0)
    data = (data - min_value)/(max_value - min_value)
    return data


def loadData(path: str, sep) -> Tuple[np.ndarray, list]:
    csv_reder = open(path, 'r', encoding='utf-8')
    data = []
    labels = []
    for line in csv_reder.readlines():
        row = line.strip().split(sep)
        data.append(row[:-1])
        labels.append(row[-1])
    data = np.array(data).astype(np.float32)
    labels = np.array(labels, dtype=np.int16)
    return data, labels


def decomp(data, dim: int) -> np.ndarray:
    """
        decompose the data to dim-2
    """
    pca = decomposition.PCA(n_components=dim)
    return pca.fit_transform(data)


def k_means(k: int, data: np.ndarray, maxIterTime: int):
    feature_dims = data.shape[1]
    prm = np.random.permutation(data.shape[0])[:k]
    mu = data[prm]
    kList = []
    [kList.append([]) for notUse in range(k)]
    for epoch in range(maxIterTime):
        # classification
        for i in range(data.shape[0]):
            xi = data[i, :]
            dist2 = np.sum((mu - xi)**2, axis=1)
            ki = np.argsort(dist2)[0]
            kList[ki].append(xi)
        for i in range(k):
            mu[i, :] = np.sum(kList[i], axis=0) / len(kList[i])
            kList[i].clear()
    return mu


def plot(path: str, data, mu):
    data = decomp(data, 2)
    mu = decomp(mu, 2)
    plt.scatter(data[:, 0], data[:, 1])
    plt.scatter(mu[:, 0], mu[:, 1], c='r')
    plt.savefig(path)


def calEntropy(mu, data, label):
    k = mu.shape[0]
    kList = []
    [kList.append([]) for notUse in range(k)]
    for i in range(data.shape[0]):
            xi = data[i, :]
            dist2 = np.sum((mu - xi)**2, axis=1)
            ki = np.argsort(dist2)[0]
            kList[ki].append(label[i])
    E = 0
    for i in range(k):
        d = {}
        num = len(kList[i])
        e = 0
        for item in kList[i]:
            if item not in d.keys():
                d[item] = 1
            else:
                d[item] += 1
        for key in d.keys():
            p = d[key]/num
            e += -p*np.log(p)
        E += e*num/data.shape[0]
    return E

if __name__ == "__main__":
    data, label = loadData("Wine.csv", sep=',')
    # for test
    # data = np.tile(data,[500,1])
    # 
    picPath = "./plt-serial.jpg"
    data = normalization(data)
    st = time.time()
    mu = k_means(3, data, 1000)
    et = time.time()
    E = calEntropy(mu,data, label)
    print("time cost: {}, entropy: {}".format(et - st, E))
    plot(picPath,data, mu)
