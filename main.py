import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, GCN2Conv, SAGEConv, GATConv, HGTConv, Linear
from torch_geometric.utils import to_undirected
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import roc_auc_score
import os
from utils import get_data
from train_model import CV_train


def set_seed(seed):
    torch.manual_seed(seed)
    #进行随机搜索的这个要注释掉
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_attr(config, param_search):
    param_grid = param_search
    param_keys = param_grid.keys()
    param_grid_list = list(ParameterGrid(param_grid))
    for param in param_grid_list:
        config.other_args = {'arg_name': [], 'arg_value': []}
        for keys in param_keys:
            setattr(config, keys, param[keys])
            config.other_args['arg_name'].append(keys)
            config.other_args['arg_value'].append(param[keys])
        yield config
    return 0

class Config:
    def __init__(self):
        self.save_file = './save_file/'

        self.kfold = 10
        self.epochs = 400
        self.each_epochs_print = 20
        self.maskMDI = False
        self.hidden_channels = 256   # 256 512
        self.num_heads = 4   # 4 8
        self.num_layers = 4   # 4 8
        self.self_encode_len = 128
        self.hidden_bias_len = 256
        self.globel_random = 100
        self.node_threshold = 0.5
        self.other_args = {'arg_name': [], 'arg_value': []}

        self.best_epoch = 4
        self.save_all_score = True
        self.save_score_file = './save_score/scores.csv'

class Data_paths:
    def __init__(self):
        self.paths = './datasets/'
        self.md = self.paths + 'm_d.csv'
        self.mm = [self.paths + 'm_fs.csv', self.paths + 'm_gs.csv', self.paths + 'm_ss.csv']
        self.dd = [self.paths + 'd_ts.csv', self.paths + 'd_gs.csv', self.paths + 'd_ss.csv']


if __name__ == '__main__':
    param_search = {
        'hidden_channels': [64,128, 256, 512],
        'num_heads': [2,4,8,16],
        'num_layers': [2,4,6,8],
        'hidden_bias_len': [64,128,256,512],

    }
    param_search = {
        'hidden_channels': [512],
        'num_heads': [8],
        'num_layers': [8],
        'hidden_bias_len': [512],

    }
    # 512.0
    # 8.0
    # 4.0
    # 256.0
    save_f = '10cvsave_data'
    params = Config()
    params = set_attr(params, param_search)
    filepath = Data_paths()

    data_list = []
    while True:
        try:
            param = next(params)
        except:
            break
        data_tuple = get_data(file_pair=filepath, params=param)
        data_idx, auc_name = CV_train(param, data_tuple)
        data_list.append(data_idx)
    if data_list.__len__()>1:
        data_all = np.concatenate(tuple(x for x in data_list), axis=1)
    else:
        data_all = data_list[0]
    np.save(param.save_file + save_f + 'save_data.npy', data_all)
    print(auc_name)

    data_idx = np.load(param.save_file + save_f + 'save_data.npy', allow_pickle=True)

    data_mean = data_idx[:, :, 2:].mean(0)
    idx_max = data_mean[:, 0].argmax()
    print()
    print('最大值为：')
    print(data_mean[idx_max, :])
    # data_all

