import numpy as np
import pandas as pd

folder = "data/"

class Splitter:
    def __init__(self, name):
        self.name = name

    def split(self):
        path = folder + self.name +".npz"
        return data_split(path)

def create_lists(indices, n_train, n_val):
    np.random.shuffle(indices)
    indices_train = indices[:n_train]
    indices_val = indices[n_train:n_train + n_val]
    indices_test = indices[n_train + n_val:]

    return indices_train, indices_val, indices_test

def data_split(path):
    raw = np.load(path, allow_pickle=True)
    key = list(raw.keys())[0]
    data = raw[key]

    in_dist = data[data[:, -1] == 0]
    out_dist = data[data[:, -1] == 1]
    n_in = in_dist.shape[0]
    n_out = out_dist.shape[0]

    in_train = int(0.6 * n_in)
    in_val = int(0.2 * n_in)

    out_train = int(0.3 * n_out)
    out_val = int(0.2 * n_out)

    in_ind = np.arange(n_in)
    out_ind = np.arange(n_out)
    in_ind_train, in_ind_val, in_ind_test = create_lists(in_ind, in_train, in_val)
    out_ind_train, out_ind_val, out_ind_test = create_lists(out_ind, out_train, out_val)

    in_dist_train = in_dist[in_ind_train]
    in_dist_val = in_dist[in_ind_val]
    in_dist_test = in_dist[in_ind_test]

    out_dist_train = out_dist[out_ind_train]
    out_dist_val = out_dist[out_ind_val]
    out_dist_test = out_dist[out_ind_test]

    val_dist = np.vstack((in_dist_val, out_dist_val))
    test_dist = np.vstack((in_dist_test, out_dist_test))
    np.random.shuffle(val_dist)
    np.random.shuffle(test_dist)

    max_n_train = min(np.random.randint(120000, 130000, 1)[0], in_dist_train.shape[0])
    max_n_val = min(np.random.randint(80000, 100000, 1)[0],val_dist.shape[0])
    max_n_contamination = min(np.random.randint(40000, 50000, 1)[0], out_dist_train.shape[0])
    max_n_test = min(np.random.randint(150000, 170000, 1)[0], test_dist.shape[0])

    in_dist_train = pd.DataFrame(in_dist_train).sample(n=max_n_train).values
    val_dist = pd.DataFrame(val_dist).sample(n=max_n_val).values
    test_dist = pd.DataFrame(test_dist).sample(n=max_n_test).values
    out_dist_train = pd.DataFrame(out_dist_train).sample(n=max_n_contamination).values

    data_dict = {key + "_train": in_dist_train,
                 key + "_val": val_dist,
                 key + "_test": test_dist,
                 key + "_contamination": out_dist_train
                 }

    return data_dict