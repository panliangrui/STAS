from torch_geometric.nn import GCNConv, SAGEConv,GATConv
import torch
import torch.nn.functional as F
import joblib
from data.dataloader import get_node,get_edge_index_image
import numpy as np
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2*out_features, 1)

    def forward(self, X, adj):
        h = self.W(X)
        a_input = torch.cat([h.repeat(1, adj.size(1), 1), h[adj]], dim=2)
        e = F.leaky_relu(self.a(a_input).squeeze(2), negative_slope=0.2)
        attention = F.softmax(e, dim=1)
        h_prime = torch.matmul(attention.unsqueeze(1), h).squeeze(1)
        return h_prime

class HGCN(torch.nn.Module):
    def __init__(self, node_features, input_size, n_hidden,output_size):
        super(HGCN, self).__init__()
        self.conv1_1 = GCNConv(in_channels=node_features, out_channels=n_hidden)
        self.conv1_2 = SAGEConv(in_channels=n_hidden, out_channels=n_hidden // 2)
        self.conv1_3 = GraphAttentionLayer(in_features=n_hidden // 2, out_features=n_hidden // 4)
        self.conv2_1 = GCNConv(in_channels=input_size, out_channels=n_hidden)
        self.conv2_2 = SAGEConv(in_channels=n_hidden, out_channels=n_hidden//2)
        self.conv2_3 = GraphAttentionLayer(in_features=n_hidden//2, out_features=n_hidden//4)
        # att_net_img_cnn = nn.Sequential(torch.nn.Linear(n_hidden, n_hidden // 2),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.LayerNorm(n_hidden // 2),
        #     torch.nn.Linear(n_hidden // 2, n_hidden // 4),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.LayerNorm(n_hidden // 4),
        #     torch.nn.Linear(n_hidden // 4, output_size))
        # self.mpool_img_swim = my_GlobalAttention(att_net_img_cnn)
        self.dropout = nn.Dropout(p=0.2)
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(n_hidden//4, n_hidden // 4),
            # torch.nn.ReLU(inplace=True),
            torch.nn.LayerNorm(n_hidden // 4),
            # torch.nn.Linear(n_hidden // 2, n_hidden // 4),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.LayerNorm(n_hidden // 4),
            torch.nn.Linear(n_hidden // 4, output_size)
            )
        self.relu = torch.nn.ReLU()


    def forward(self, x, x1, edge_index, edge_index1):
        '''
        GCN
        '''
        x = self.conv1_1(x, edge_index)
        x0 = self.relu(x)
        x1 = self.conv2_1(x1, edge_index1)
        x1_0 = self.relu(x1)

        x0 = x0+x1_0
        x = self.conv1_2(x0, edge_index)
        x = self.conv1_3(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.MLP(x)
        x = torch.squeeze(x)#.mean(0)


        x1_0 = x1_0+x1
        x1 = self.conv2_2(x1_0,edge_index1)
        x1 = self.conv2_3(x1, edge_index1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        # x1 = self.MLP(x1)
        # x1 = self.dropout(x1)
        # batch_swim = torch.zeros(len(x1), dtype=torch.long).to(device)
        # _, x1 = self.mpool_img_swim(x1, batch_swim)
        # x1 = torch.mean(x1,dim=1)
        x1 = self.MLP(x1)
        x1 = torch.squeeze(x1)  # .mean(0)

        concatenated_feature = torch.cat((x, x1), dim=0).unsqueeze(0)#x1.unsqueeze(0)#concatenated_feature = x1.unsqueeze(0)#
        # self.fc = nn.Linear(len(concatenated_feature), 1).to(device)
        # x = self.fc(concatenated_feature)
        out = torch.mean(concatenated_feature)#nn.Sigmoid()(torch.mean(concatenated_feature))
        out = torch.unsqueeze(out, 0).unsqueeze(1)
        # x = torch.mean(x)
        return out






