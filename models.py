from torch_geometric.nn import GCNConv, GATConv
from graphmask_models import GM_GCNconv as GCNConv
from graphmask_models import GM_GATConv as GATConv
import torch.nn as nn
import torch.nn.functional as F
import torch
from math import log
from layer import GCN2Conv
import math
import torch_geometric.nn as gnn


class GNNPool(nn.Module):
    def __init__(self):
        super().__init__()

class GlobalMeanPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return gnn.global_mean_pool(x, batch)


class IdenticalPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return x


class GCN_Encoder(nn.Module):
    def __init__(self, in_dim, hid_dim, num_classes, num_layers, dropout = 0):
        super(GCN_Encoder, self).__init__()
        self.convs = nn.ModuleList()
        self.BNs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, 2*hid_dim))
        # self.BNs.append(nn.BatchNorm1d(2*hid_dim))
        for _ in range(1,num_layers - 1):
            self.convs.append(GCNConv(2*hid_dim, 2*hid_dim))
            # self.BNs.append(nn.BatchNorm1d(2*hid_dim))
        self.convs.append(GCNConv(2*hid_dim, hid_dim))
        # self.BNs.append(nn.BatchNorm1d(hid_dim))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim, num_classes)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = self.dropout(x)
            x = self.activation(conv(x, edge_index))
            # x = bn(x)
        # x = self.dropout(x)
        x = self.fc(x)
        return x

    def embed(self, x, edge_index):
        for conv in self.convs:
            x = self.activation(conv(x, edge_index))
        return x


class Projection_Head(nn.Module):
    def __init__(self, hid_dim, proj_hid_dim, proj_dim, type = 'Linear', dropout = 0):
        super(Projection_Head, self).__init__()
        self.fcs = nn.ModuleList()
        if type == 'Linear':
            self.fcs.append(nn.Linear(hid_dim, proj_dim,dropout))
        else:
            self.fcs.append(nn.Sequential(nn.Linear(hid_dim, proj_hid_dim, dropout), nn.BatchNorm1d(proj_hid_dim)))
            self.fcs.append(nn.Linear(proj_hid_dim, proj_dim, dropout))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0)



    def forward(self,x):
        for fc in self.fcs:
            x = fc(self.dropout(x))
            x = self.activation(x)

        return x

class SupCG(nn.Module):
    def __init__(self, in_dim, hid_dim, proj_hid_dim, proj_dim, num_classes, num_layers, type = 'Linear', dropout = 0):
        super(SupCG, self).__init__()
        self.encoder = GCN_Encoder(in_dim, hid_dim, num_classes, num_layers, dropout)
        self.projection = Projection_Head(hid_dim, proj_hid_dim, proj_dim, type, dropout)


    def forward(self, x, edge_index1, rand_x1 = None, rand_x2 = None, edge_index2 = None, edge_index3 = None, GCN2 = False):
        if not GCN2:
            if edge_index2 is not None and edge_index3 is not None:
                x1 = self.encoder.embed(x, edge_index1)
                x1 = self.projection(x1)
                x1 = F.normalize(x1, dim=1)

                x2 = self.encoder.embed(rand_x1, edge_index2)
                x2 = self.projection(x2)
                x2 = F.normalize(x2, dim=1)

                x3 = self.encoder.embed(rand_x2, edge_index3)
                x3 = self.projection(x3)
                x3 = F.normalize(x3, dim=1)

                return x1, x2, x3
            elif edge_index2 is not None:
                x1 = self.encoder.embed(x, edge_index1)
                x1 = self.projection(x1)
                x1 = F.normalize(x1, dim=1)

                x2 = self.encoder.embed(rand_x1, edge_index2)
                x2 = self.projection(x2)
                x2 = F.normalize(x2, dim=1)

                return x1, x2

            else:
                x1 = self.encoder.embed(x, edge_index1)
                x1 = self.projection(x1)
                x1 = F.normalize(x1, dim=1)
                return x1
        else:
            if edge_index2 is not None and edge_index3 is not None:
                x1 = self.encoder.embed(x, edge_index1, 0.1, 0.4)
                x1 = self.projection(x1)
                x1 = F.normalize(x1, dim=1)

                x2 = self.encoder.embed(rand_x1, edge_index2, 0.1, 0.4)
                x2 = self.projection(x2)
                x2 = F.normalize(x2, dim=1)

                x3 = self.encoder.embed(rand_x2, edge_index3, 0.1, 0.4)
                x3 = self.projection(x3)
                x3 = F.normalize(x3, dim=1)

                return x1, x2, x3
            elif edge_index2 is not None:
                x1 = self.encoder.embed(x, edge_index1, 0.1, 0.4)
                x1 = self.projection(x1)
                x1 = F.normalize(x1, dim=1)

                x2 = self.encoder.embed(rand_x1, edge_index2,  0.1, 0.4)
                x2 = self.projection(x2)
                x2 = F.normalize(x2, dim=1)

                return x1, x2

            else:
                x1 = self.encoder.embed(x, edge_index1, 0.1, 0.4)
                x1 = self.projection(x1)
                x1 = F.normalize(x1, dim=1)
                return x1

class LinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self,x):
        return self.fc(x)

class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim,out_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
    def forward(self,x):
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Teacher(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_class,
                 base_model=GCNConv, k: int = 2, dropout = 0.7):
        super(Teacher, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]

        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.fc = nn.Linear(out_channels, num_class)

    def forward(self, x, edge_index):
        for i in range(self.k):
            x = self.dropout(x)
            x = self.activation(self.conv[i](x, edge_index))
        x = self.fc(x)
        return x

    def embed(self,x, edge_index):
        for i in range(self.k):
            x = self.dropout(x)
            x = self.activation(self.conv[i](x, edge_index))
        return x

class GATTeacher(nn.Module):
    def __init__(self,in_channels: int, out_channels: int, num_class,
                 base_model=GCNConv, k: int = 2, heads = 8, dropout = 0.6):
        super(GATTeacher, self).__init__()
        self.conv = [GATConv(in_channels, 2 * out_channels, heads = heads, negative_slope = 0.1, dropout = dropout)]
        self.k = k

        self.conv.append(GATConv(16 * out_channels, out_channels, heads = 8, negative_slope = 0.1, dropout = dropout))
        self.conv = nn.ModuleList(self.conv)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.fc = nn.Linear(8*out_channels, num_class)
    def forward(self, x, edge_index):
        for i in range(self.k):
            x = self.dropout(x)
            x = self.activation(self.conv[i](x, edge_index))
        x = self.fc(x)
        return x


class GCN2(nn.Module):
    def __init__(self, model_level, dim_node, dim_hidden, num_classes, alpha, theta, num_layers, shared_weights,dropout = 0):
        super().__init__()

        convs = []
        for i in range( num_layers ):
            convs.append(GCN2Conv(dim_hidden, alpha, theta, i + 1, shared_weights = shared_weights))
        self.convs = nn.ModuleList(
            convs
        )
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(dim_node, dim_hidden))
        self.fcs.append(nn.Linear(dim_hidden, num_classes))
        self.relu = nn.ReLU()
        self.dropout = dropout
        if model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()

    def forward(self, x, edge_index, batch = None):
        """
        :param Required[data]: Batch - input data
        :return:
        """
        # x, edge_index = data.x, data.edge_index
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.relu(self.fcs[0](x))
        x_0 = x
        for i,conv in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.relu(conv(x, x_0, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        out = self.fcs[-1](x)

        return out


    def get_emb(self,*args):
        if len(args) == 1:
            x, edge_index = args[0].x, args[0].edge_index
        else:
            x, edge_index = args[0], args[1]
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.relu(self.fcs[0](x))
        x_0 = x
        for i,conv in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.relu(conv(x, x_0, edge_index))
        return x
