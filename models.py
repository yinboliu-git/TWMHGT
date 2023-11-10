import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, GCN2Conv, SAGEConv, GATConv, HGTConv, Linear,HANConv
from torch_geometric.utils import to_undirected
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from globel_args import device
# HGTConv = HANConv


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads, num_layers, data, params):

        super().__init__()
        self.params = params
        self.lin_dict = torch.nn.ModuleDict()
        self.lin_xe_dict = torch.nn.ModuleDict()
        self.att_dict = {}
        for node_type in data.node_types:
            # self.lin_dict[node_type] = Linear(-1, hidden_channels) # -1表示根据输入的数据的维度自动调整
            self.lin_xe_dict[node_type] = Linear(-1, hidden_channels) # -1表示根据输入的数据的维度自动调整
            self.att_dict[node_type] = Parameter(torch.ones((num_layers,)), requires_grad=True)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            #  in_channels: Union[int, Dict[str, int]],
            conv = HGTConv(-1, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)


        self.x_bias = {'n1':torch.randn((data['edge_m'].shape[0],params.hidden_bias_len),dtype=torch.float32, device=device),
                       'n2':torch.randn((data['edge_m'].shape[1],params.hidden_bias_len),dtype=torch.float32, device=device)}

        self.lin_dict_bias = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict_bias[node_type] = Linear(-1, hidden_channels) # -1表示根据输入的数据的维度自动调整

        self.convs_bias = torch.nn.ModuleList()
        for _ in range(num_layers):
            #  in_channels: Union[int, Dict[str, int]],
            conv = HGTConv(-1, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs_bias.append(conv)

        self.fc = Linear(hidden_channels*2, 2)
        self.dropout = torch.nn.Dropout(0.5)
        self.att = Parameter(torch.ones((2,)), requires_grad=True)
        self.att_bias = Parameter(torch.ones((2,)), requires_grad=True)


    def forward(self, x_dict_, edge_index_dict, edge_index, xe_):
        x_dict = x_dict_.copy()
        # xe_ = None
        if xe_ != None:
            xe = xe_.copy()
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()
            if xe_ != None:
                xe[node_type] = self.lin_xe_dict[node_type](xe[node_type][0]).relu_()
                x_dict[node_type] = 0 * x_dict[node_type] + self.att[1] * xe[node_type]

        all_list = []
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            all_list.append(x_dict.copy())


        x_dict_bias = x_dict_.copy()
        for node_type, x in x_dict_bias.items():
            x_dict_bias[node_type] = self.lin_dict_bias[node_type](self.x_bias[node_type]).relu_()

        all_list_bias = []
        for conv in self.convs_bias:
            x_dict_bias = conv(x_dict_bias, edge_index_dict)
            all_list_bias.append(x_dict_bias.copy())

        for i,_ in x_dict_.items():
            x_dict[i] = self.att_bias[0]*torch.cat(tuple(x[i] for x in all_list), dim=1) + \
                        self.att_bias[1]*torch.cat(tuple(x[i] for x in all_list_bias), dim=1)


        m_index = edge_index[0]
        d_index = edge_index[1]
        #
        Em = self.dropout(x_dict['n1'])
        Ed = self.dropout(x_dict['n2'])
        y = Em@Ed.t()
        # y = torch.cat((Em, Ed), dim=1)
        # y = self.fc(y)
        y_all = y
        y = y[m_index,d_index].unsqueeze(-1)
        return y, y_all

