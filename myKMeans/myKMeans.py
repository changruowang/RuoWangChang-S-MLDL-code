# --*-- coding:utf-8 --*--
import numpy as np

class MyKMeansPlus(object):
    def __init__(self, maxIter=500):
        self.maxIter = maxIter

    def fit(self,samples, k):
        samples = np.array(samples)
        m, n = samples.shape

        def getCenters(k_):
            distTmp = np.zeros(m)
            centers = []
            centers.append(samples[np.random.randint(m)])

            while len(centers) < k_:
                for j in range(m):
                    distTmp[j] = np.min(np.linalg.norm(samples[j, :] - np.array(centers), axis=1))
                distSumRand = np.sum(distTmp) * np.random.rand()
                for j in range(m):
                    distSumRand -= distTmp[j]
                    if distSumRand < 0:
                        centers.append(samples[j, :])
                        break
            return np.array(centers)

        centersOld = getCenters(k)
        centersNew = np.zeros_like(centersOld)
        labels = np.zeros(m, dtype=np.int8)
        cnt = 0
        while cnt < self.maxIter:
            cnt += 1
            for i in range(m):   #根据中心点重新给样本划分族
                labels[i] = np.argmin(np.linalg.norm(samples[i, :] - centersOld, axis=1))

            for i in range(k):   #更新中心点
                centersNew[i] = np.mean(samples[labels==i,:], axis=0)

            print(centersOld, '\n',centersNew)
            if np.sum(np.fabs(centersNew-centersOld)) < 0.0001: break

            centersOld = centersNew

        print(cnt)
        return labels