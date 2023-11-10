import numpy as np


data = np.load('10cvsave_data.npy', allow_pickle=True)
a = data[:,:,2:].mean(0)
a[np.where(a[:,0].max()==a[:,0])[0]]
print()


