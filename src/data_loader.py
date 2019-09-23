import numpy as np
from torch.utils.data import Dataset

class ToyDataset(Dataset):
    def __init__(self, seed, size, n_pools, samples, input_size, dist, custom=None):
        """
        Returns a random toy dataset of indices
        Args:
            seed: (int) seed for numpy
            size: (int) determines the number of unique indices in the total dataset,
                it should be divisible by input size
            n_pools: (int) hidden number of clusters for the loss function
            samples: (int) number of samples for each input
            input_size: (int) input size of the neural network
            dist: (str) determines the size distribution of the pools, can be uniform, exp, perm, or custom
            custom: (list of float) ignored except if dist is set to custom
        """
        self.size =  input_size * size * samples
        self.length = size * samples
        np.random.seed(seed)
        # generating data
        self.data = np.empty(shape=size, dtype=np.int)
        self.data_group = np.empty(shape=size, dtype=np.int)
        self.data[:] = np.arange(0,size)

        # generating group membership
        s = np.random.uniform(0,1,size)
        if dist == 'uniform':
            for i in range(0,n_pools):
                cond = (i+1)/n_pools
                filt = np.where(s<cond)[0]
                self.data_group[filt] = i
                s[filt] = 1
        elif dist == 'exp':
            cond = 0
            for i in range(0,n_pools):
                cond += 1/2**(i+1)
                filt = np.where(s<cond)[0]
                self.data_group[filt] = i
                s[filt] = 1
        elif dist == 'custom':
            cust = [x/sum(custom) for x in custom]
            cond = 0
            for i in range(0,n_pools):
                cond += cust[i]
                filt = np.where(s<cond)[0]
                self.data_group[filt] = i
                s[filt] = 1
        elif dist == 'perm':
            samples = 1
        else:
            raise NameError('Invalid input for dist, see class definition!')

        self.data = self.data.reshape((size//input_size, input_size))
        self.data_group = self.data_group.reshape((size//input_size, input_size))

        # sampling
        self.xdata = np.empty(shape=(samples* size//input_size, input_size), dtype=np.int)
        self.ydata = np.empty(shape=(samples* size//input_size, input_size), dtype=np.int)

        if dist != 'perm':
            for i in range(0,size//input_size):
                self.xdata[samples*i:(i+1)*samples, :] = self.data[i, :]
                for j in range(0, n_pools):
                    ind =  np.where(self.data_group[i, :] == j)[0]
                    sample_space = self.data[i, ind] - self.data[i, 0]
                    for k in range(0, ind.shape[0]):
                        self.ydata[i*samples:(i+1)*samples, ind[k]] = np.random.choice(sample_space, samples)
        else:
            self.xdata = self.data
            for i in range(0,size//input_size):
                self.ydata[i, :] = np.random.permutation(self.data[i, :])- self.data[i, 0]




    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return self.xdata[item, :], self.ydata[item, :]

    def get_groups(self):
        return self.data_group

    def get_x(self):
        return self.xdata

    def get_y(self):
        return self.ydata


# For testing only
if __name__ == '__main__':
    toy = ToyDataset(1,1024,10,10,128,'uniform')
    print(toy.__getitem__(10))

