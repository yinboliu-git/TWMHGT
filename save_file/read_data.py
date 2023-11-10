import numpy as np
data_idx = np.load('./5折交叉验证全部数据.npy', allow_pickle=True)

print(data_idx.shape())
data_mean = data_idx[:, :, 2:].mean(0)
idx_max = data_mean[:, 0].argmax()
print()
print('最大值为：')
print(data_mean[idx_max, :])
# data_all
