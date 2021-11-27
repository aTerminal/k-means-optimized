import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np
import csv
import numpy as np
from numpy.core.fromnumeric import shape
from sklearn import decomposition
import time
from mpi4py import MPI


def normalization(data: np.ndarray) -> np.ndarray:
    max_value = np.max(data, axis=0)
    min_value = np.min(data, axis=0)
    data = (data - min_value)/(max_value - min_value)
    return data


def loadData(path: str) -> Tuple[np.ndarray, list]:
    csv_reder = csv.reader(open(path, encoding='utf-8'))
    data = []
    labels = []
    for row in csv_reder:
        data.append(row[:-1])
        labels.append(row[-1])
    data = np.array(data).astype(np.float32)
    labels = np.array(labels)
    return data, labels


def decomp(data, dim: int) -> np.ndarray:
    pca = decomposition.PCA(n_components=dim)
    return pca.fit_transform(data)


def k_means(k: int, data: np.ndarray, maxIterTime: int, hyperparameter: int):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if rank == 0:
        prm = np.random.permutation(data.shape[0])[:k]
        mu = data[prm]
    else:
        mu = np.empty(shape=[k, data.shape[1]], dtype=data.dtype)

    kList = []
    [kList.append([]) for notUse in range(k+1)]

    size_shape0 = data.shape[0] // size + 1 if data.shape[0] % size != 0 else data.shape[0] // size
    displ = []
    sum = 0

    for epoch in range(size):
        displ.append(sum)
        sum = sum + size_shape0
        pass

    if rank < size-1:
        recv_obj = data[displ[rank]:displ[rank+1], :].copy()
        save_buf = [[], []]
    elif rank == size-1:
        recv_obj = data[displ[rank]:, :].copy()
        save_buf = [[], []]

    for i in range(k):
        kList[i].append(np.zeros(data.shape[1]))
    [kList[k].append(0) for notUse in range(k)]

    dic = {}
    for seq, arr in enumerate(recv_obj):
        dic[arr.tobytes()] = []

    for epoch in range(maxIterTime):
        comm.Bcast(mu, root=0)
        if len(list(dic.items())) != 0 and len(list(dic.items())[0][1]) >= hyperparameter:
            for item in dic.items():
                flag = 0
                r_cmp = item[1][:-(hyperparameter+1):-1]
                if len(r_cmp) < hyperparameter:
                    continue
                for i in range(1, hyperparameter):
                    if r_cmp[0] != r_cmp[i]:
                        flag = 1
                        break
                if flag == 1:
                    continue
                else:
                    ar = np.where((recv_obj == np.frombuffer(item[0], dtype=data.dtype)).all(axis=1))
                    recv_obj = np.delete(recv_obj, ar, axis=0)
                    save_buf[0].append(np.frombuffer(item[0], dtype=data.dtype))
                    save_buf[1].append(r_cmp[0])
            for i in save_buf[0]:
                if dic.__contains__(i.tobytes()):
                    dic.pop(i.tobytes())
        
        for i in range(len(save_buf[1])):
            kList[save_buf[1][i]][0] += save_buf[0][i]
            kList[k][save_buf[1][i]] += 1
            pass

        for i in range(recv_obj.shape[0]):
            xi = recv_obj[i, :]
            dist2 = np.sum((mu - xi)**2, axis=1)
            ki = np.argsort(dist2)[0]
            kList[ki][0] += xi
            kList[k][ki] += 1
            dic[xi.tobytes()].append(ki)

        kList_g = comm.gather(kList, root=0)
        if rank == 0:
            j = 1
            while j < size:
                i = 0
                while i < k:
                    kList[i][0] += kList_g[j][i][0]
                    i += 1
                r = 0
                while r < k:
                    kList[k][r] += kList_g[j][k][r]
                    r += 1
                j += 1
            for i in range(k):
                if kList[k][i] != 0:
                    mu[i, :] = kList[i][0] / kList[k][i]
        
        for ep in range(k):
            kList[ep].clear()
            kList[ep].append(np.zeros(data.shape[1]))
            kList[k][ep] = 0

    return mu, kList

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

def plot(path:str, data, mu):
    data = decomp(data, 2)
    mv = decomp(mu, 2)
    plt.scatter(data[:, 0], data[:, 1])
    plt.scatter(mv[:, 0], mv[:, 1], c='r')
    plt.savefig(path)

def predict():
    pass

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    hyperparameter = 5
    data, label = loadData("./Wine.csv")
    data = normalization(data)
    st = time.time()
    mu, kList = k_means(3, data, 1000, hyperparameter)
    et = time.time()
    if rank == 0:
        plot("./plt.jpg", data, mu)
        E = calEntropy(mu, data, label)
        print("time cost: {}, entropy: {}".format(et - st, E))
