from torch_geometric.nn import GCNConv, SAGEConv,GATConv
import torch
import torch.nn.functional as F
import joblib
from data.dataloader import get_node,get_edge_index_image
import numpy as np
import torch.nn as nn
from models.kan import KAN


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# image_path_cnn_fea = joblib.load('P:\\xiangya2\\STAS\\stas_all\\256\\np\\1199003-1-N.pkl')
# image_path_swim_fea = joblib.load('P:\\xiangya2\\STAS\\stas_all\\256\\np_swim\\1199003-1-N.pkl')
# node_image_path_cnn_fea, node_image_path_swim_fea =get_node(image_path_cnn_fea, image_path_swim_fea)
#
# edge_index_image_cnn = torch.tensor(get_edge_index_image(image_path_cnn_fea), dtype=torch.long)
# edge_index_image_swim = torch.tensor(get_edge_index_image(image_path_swim_fea), dtype=torch.long)
# node_image_path_cnn_fea = torch.Tensor(np.stack(node_image_path_cnn_fea))
# node_image_path_swim_fea = torch.Tensor(np.stack(node_image_path_swim_fea))
# from sklearn.decomposition import PCA
# pca1 = PCA(n_components=100)
# node_image_path_cnn_fea = torch.Tensor(pca1.fit_transform(node_image_path_cnn_fea))
# from sklearn.decomposition import PCA
# pca2 = PCA(n_components=100)
# node_image_path_swim_fea = torch.Tensor(pca2.fit_transform(node_image_path_swim_fea))
from models.multiview import fusion_model_graph,my_GlobalAttention#, GCN


class kanGCN(torch.nn.Module):
    def __init__(self, node_features, input_size, n_hidden,output_size):
        super(kanGCN, self).__init__()
        self.conv1_1 = GCNConv(in_channels=node_features, out_channels=n_hidden)
        self.conv1_2 = SAGEConv(in_channels=n_hidden, out_channels=n_hidden // 2)
        self.conv2_1 = GCNConv(in_channels=input_size, out_channels=n_hidden)
        self.conv2_2 = SAGEConv(in_channels=n_hidden, out_channels=n_hidden//2)
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
            torch.nn.Linear(n_hidden//2, n_hidden // 2),
            # torch.nn.ReLU(inplace=True),
            torch.nn.LayerNorm(n_hidden // 2),
            # torch.nn.Linear(n_hidden // 2, n_hidden // 4),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.LayerNorm(n_hidden // 4),
            torch.nn.Linear(n_hidden // 2, output_size)
            )
        self.KAN = KAN([n_hidden//2, n_hidden//2, 1])
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
        x = self.relu(x)
        x = self.dropout(x)
        x = self.KAN(x)
        x = torch.squeeze(x)#.mean(0)


        x1_0 = x1_0+x1
        x1 = self.conv2_2(x1_0,edge_index1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        # x1 = self.MLP(x1)
        # x1 = self.dropout(x1)
        # batch_swim = torch.zeros(len(x1), dtype=torch.long).to(device)
        # _, x1 = self.mpool_img_swim(x1, batch_swim)
        # x1 = torch.mean(x1,dim=1)
        x1 = self.KAN(x1)
        x1 = torch.squeeze(x1)  # .mean(0)

        concatenated_feature = torch.cat((x, x1), dim=0).unsqueeze(0)#x1.unsqueeze(0)#concatenated_feature = x1.unsqueeze(0)#
        # self.fc = nn.Linear(len(concatenated_feature), 1).to(device)
        # x = self.fc(concatenated_feature)
        out = torch.mean(concatenated_feature)#nn.Sigmoid()(torch.mean(concatenated_feature))
        out = torch.unsqueeze(out, 0).unsqueeze(1)
        # x = torch.mean(x)
        return out



#
# model = kanGCN(node_features=100, input_size=100, n_hidden=60, output_size=1).to(device)
#
# out = model(node_image_path_cnn_fea.to(device), node_image_path_swim_fea.to(device),edge_index_image_cnn.to(device), edge_index_image_swim.to(device))
# print(out)
#
# # if 'N' in id:
# #     label = 0
# # elif 'P' in id:
# #     label = 1
# # else:
# #     continue
# label = 0
# label = torch.Tensor([int(label)]).to(device)
# # label = torch.tensor(label)# label = torch.Tensor([int(label)]).to(device)
# label = label.unsqueeze(1)
# criterion = torch.nn.BCEWithLogitsLoss()
# loss = criterion(out, label)
# print(loss)
# prob = torch.sigmoid(out)
# print(prob)










#
#
#
# model1 = fusion_model_graph(args=None, in_feats_cnn=100, in_feats_swim=100, n_hidden=50, out_classes=50, dropout=0.2, num_classes=1).to(device)
#
# out = model1(node_image_path_cnn_fea.to(device), node_image_path_swim_fea.to(device), edge_index_image_cnn.to(device), edge_index_image_swim.to(device))
# print(out)
