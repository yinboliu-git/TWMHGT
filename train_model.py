import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, GCN2Conv, SAGEConv, GATConv, HGTConv, Linear
from torch_geometric.utils import to_undirected
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from models import HGT
from globel_args import device
from utils import get_metrics
from sklearn.model_selection import KFold



def trian_model(data,y, edg_index_all, train_idx, test_idx, param):
    hidden_channels, num_heads, num_layers = (
        param.hidden_channels, param.num_heads, param.num_layers,
    )

    epoch_param = param.epochs

    # 模型构建
    model = HGT(hidden_channels, num_heads=num_heads, num_layers=num_layers, data=data, params=param).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0002)

    # 训练模型
    auc_list = []
    model.train()
    for epoch in range(1, epoch_param+1):
        optimizer.zero_grad()
        out = model(data['x_dict'], data['edge_dict'],
                    edge_index=edg_index_all.to(device), xe_ = data['xe'])
        # 使用train数据进行训练
        # y_one_hot = F.one_hot(y[train_idx].long().squeeze(), num_classes=2).float().to(device)
        loss = F.binary_cross_entropy_with_logits(out[train_idx].to(device), y[train_idx].to(device))
        loss.backward()
        optimizer.step()
        loss = loss.item()
        if epoch % param.each_epochs_print == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            # 模型验证
            model.eval()
            with torch.no_grad():
                # 获得所有数据
                out,y_all = model(data['x_dict'], data['edge_dict'],
                            edge_index=edg_index_all, xe_ = data['xe'])
                # 提取验证集数据
                out_pred_s = out[test_idx].to('cpu').detach().numpy()
                out_pred = out_pred_s
                y_true = y[test_idx].to('cpu').detach().numpy()
                # 计算AUC
                auc = roc_auc_score(y_true, out_pred)
                print('AUC:', auc)

                # 计算所有评价指标
                auc_idx, auc_name = get_metrics(y_true, out_pred)
                auc_idx.extend(param.other_args['arg_value'])
                auc_idx.append(epoch)
            auc_list.append(auc_idx)
            model.train()
        if param.save_all_score & epoch == param.best_epoch:
            y_all = pd.DataFrame(y_all)
            y_all.to_csv(param.save_score_file, header=None)
            raise OSError('save_score_file')

    auc_name.extend(param.other_args['arg_name'])
    return auc_list, auc_name


def CV_train(param, args_tuple=()):
    data, y, edg_index_all = args_tuple
    idx = np.arange(y.shape[0])
    k_number = 1
    k_fold = param.kfold
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=param.globel_random)

    kf_auc_list = []
    for train_idx,test_idx  in kf.split(idx):
        print(f'正在运行第{k_number}折, 共{k_fold}折...')
        auc_idx, auc_name = trian_model(data, y, edg_index_all, train_idx, test_idx, param)
        k_number += 1

        kf_auc_list.append(auc_idx)

    data_idx = np.array(kf_auc_list)
    return data_idx, auc_name


