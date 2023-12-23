import numpy as np
from sklearn.utils.extmath import randomized_svd

dic = np.load("features-gabor.npy")
k = 100
n = dic.shape[0]
t = dic.shape[1]
print(dic.shape)

# Compress the dictionary using randomized SVD
U, Sigma, Vt = randomized_svd(dic, n_components=k, n_iter='auto', random_state=1)

Dk = dic @ Vt.T

np.savez("svd-BW-tiles.npz", Dk=Dk, U=U, Sigma=Sigma, Vt=Vt)

