import os
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.nn import SAGEConv, LayerNorm
import torch

from torch_geometric.nn import GATConv,GCNConv,GENConv, GINConv,GMMConv,GPSConv, GINEConv, GATv2Conv, APPNP, GatedGraphConv, ARMAConv


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F

def GNN_relu_Block(dim2, dropout=0.1):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(
        # GATConv(in_channels=1024,out_channels=512),
        # nn.Linear(1024, 512),
        nn.ReLU(),
        LayerNorm(dim2),
        nn.Dropout(p=dropout))

class GraphARMAConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_stacks, num_layers):
        super(GraphARMAConv, self).__init__()

        self.armac = nn.ModuleList()
        self.armac.append(ARMAConv(input_dim, hidden_dim, num_layers))
        for _ in range(num_stacks - 1):
            self.armac.append(ARMAConv(hidden_dim, hidden_dim, num_layers))
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        for armac in self.armac:
            x = armac(x, edge_index)
        x = self.fc(x)
        return x

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

class my_GlobalAttention(torch.nn.Module):
    def __init__(self, gate_nn, nn=None):
        super(my_GlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, x, batch, size=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        return out, gate

    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.nn)


def Mix_mlp(dim1):
    return nn.Sequential(
        nn.Linear(dim1, dim1),
        nn.GELU(),
        nn.Linear(dim1, dim1))

class MixerBlock(nn.Module):
    def __init__(self, dim1, dim2):
        super(MixerBlock, self).__init__()

        self.norm = LayerNorm(dim1)
        self.mix_mip_1 = Mix_mlp(dim1)
        self.mix_mip_2 = Mix_mlp(dim2)

    def forward(self, x):
        x = x.transpose(0, 1)
        # z = nn.Linear(512, 3)(x)

        y = self.norm(x)
        # y = y.transpose(0,1)
        y = self.mix_mip_1(y)
        # y = y.transpose(0,1)
        x = x + y
        y = self.norm(x)
        y = y.transpose(0, 1)
        z = self.mix_mip_2(y)
        z = z.transpose(0, 1)
        x = x + z
        x = x.transpose(0, 1)

        # y = self.norm(x)
        # y = y.transpose(0,1)
        # y = self.mix_mip_1(y)
        # y = y.transpose(0,1)
        # x = self.norm(y)
        return x



from torch_geometric.nn import GraphConv
class fusion_model_our(nn.Module):
    def __init__(self, args, in_feats_cnn, in_feats_swim, n_hidden, out_classes, dropout=0.1, num_classes=3):
        super(fusion_model_our, self).__init__()

        self.img_gnn_cnn = SAGEConv(in_channels=in_feats_cnn, out_channels=out_classes)#args, 2, 1024
        # self.img_gnn_cnn = GATConv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_cnn = GCNConv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_cnn = GENConv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_cnn = GINNet(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_cnn = GPSNet(in_channels=in_feats, out_channels=out_classes, conv=None)
        # self.img_gnn_cnn = GATv2Conv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_cnn = GraphARMAConv(input_dim=in_feats, hidden_dim=256, output_dim=out_classes, num_stacks=2, num_layers=2)
        # self.img_gnn_cnn = PreModel(args, 2, 1024)
        self.img_relu_cnn = GNN_relu_Block(out_classes)

        self.img_gnn_swim = SAGEConv(in_channels=in_feats_swim, out_channels=out_classes)  # args, 2, 1024
        # self.img_gnn_swim = GATConv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_swim = GCNConv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_swim = GENConv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_swim = GINNet(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_swim = GPSNet(in_channels=in_feats, out_channels=out_classes, conv=None)
        # self.img_gnn_swim = GATv2Conv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_swim = GraphARMAConv(input_dim=in_feats, hidden_dim=256, output_dim=out_classes, num_stacks=2, num_layers=2)
        # self.img_gnn_swim = PreModel(args, 2, 1024)
        self.img_relu_swim = GNN_relu_Block(out_classes)

        #TransformerConv

        att_net_img_cnn = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(), nn.Linear(out_classes // 4, 1))
        self.mpool_img_cnn = my_GlobalAttention(att_net_img_cnn)

        att_net_img_swim = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(), nn.Linear(out_classes // 4, 1))
        self.mpool_img_swim = my_GlobalAttention(att_net_img_swim)

        # att_net_img_512 = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(), nn.Linear(out_classes // 4, 1))
        # self.mpool_img_512 = my_GlobalAttention(att_net_img_512)



        # self.mae = PretrainVisionTransformer(encoder_embed_dim=out_classes, decoder_num_classes=out_classes, decoder_embed_dim=out_classes, encoder_depth=1, decoder_depth=1, train_type_num=train_type_num)
        # self.mae = ConvNeXtV2(depths=[2, 2, 6, 6], dims=[512, 512, 512, 512])

        # self.mix = MixerBlock(train_type_num, out_classes)

        self.lin1_img_cnn = torch.nn.Linear(out_classes, out_classes // 4)
        self.lin1_img_swim = torch.nn.Linear(out_classes, out_classes // 4)
        # self.lin1_img_512 = torch.nn.Linear(out_classes, out_classes // 4)
        self.lin2_img_cnn = torch.nn.Linear(out_classes // 4, 4)
        self.lin2_img_swim = torch.nn.Linear(out_classes // 4, 4)
        # self.lin2_img_512 = torch.nn.Linear(out_classes // 4, 4)

        # self.lin2_rna = torch.nn.Linear(out_classes // 4, 1)
        # self.lin1_cli = torch.nn.Linear(out_classes, out_classes // 4)
        # self.lin2_cli = torch.nn.Linear(out_classes // 4, 1)

        self.norm_img_cnn = LayerNorm(out_classes // 4)
        self.norm_img_swim = LayerNorm(out_classes // 4)
        # self.norm_img_512 = LayerNorm(out_classes // 4)
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(out_classes // 4 * 2, num_classes)
        # self.fc1 = nn.Linear(n_hidden, n_hidden)
        # self.fc2 = nn.Linear(n_hidden, 4)
        # self.conv2 = GraphConv(n_hidden, 4)

    def forward(self, node_image_path_cnn_fea, node_image_path_swim_fea, edge_index_image_cnn, edge_index_image_swim):
        att_2 = []
        pool_x = torch.empty((0)).to(device)
        x_img_cnn = self.img_gnn_cnn(node_image_path_cnn_fea, edge_index_image_cnn)#loss_img,
        x_img_cnn = torch.relu(x_img_cnn)#self.img_relu_128(x_img_128)
        batch_cnn = torch.zeros(len(x_img_cnn), dtype=torch.long).to(device)
        pool_x_img_cnn, att_img_cnn = self.mpool_img_cnn(x_img_cnn, batch_cnn)
        att_2.append(att_img_cnn)
        pool_x_img_cnn = torch.cat((pool_x, pool_x_img_cnn), 0)

        x_img_swim = self.img_gnn_swim(node_image_path_swim_fea, edge_index_image_swim)  # loss_img,
        x_img_swim = self.img_relu_swim(x_img_swim)
        batch_swim = torch.zeros(len(x_img_swim), dtype=torch.long).to(device)
        pool_x_img_swim, att_img_swim = self.mpool_img_swim(x_img_swim, batch_swim)
        att_2.append(att_img_swim)
        pool_x_img_swim = torch.cat((pool_x_img_cnn, pool_x_img_swim), 0)


        ##跳跃连接
        x_img_cnn = x_img_cnn + pool_x_img_cnn[0]
        x_img_swim = x_img_swim + pool_x_img_swim[1]
        # x_img_512 = x_img_512 + pool_x_img_512[2]

        att_3 = []
        pool_x = torch.empty((0)).to(device)
        batch_cnn = torch.zeros(len(x_img_cnn), dtype=torch.long).to(device)
        pool_x_img_cnn, att_img_cnn = self.mpool_img_cnn(x_img_cnn, batch_cnn)
        att_3.append(att_img_cnn)
        pool_x_cnn = torch.cat((pool_x, pool_x_img_cnn), 0)

        batch_swim = torch.zeros(len(x_img_swim), dtype=torch.long).to(device)
        pool_x_img_swim, att_img_swim = self.mpool_img_swim(x_img_swim, batch_swim)
        att_3.append(att_img_swim)
        pool_x_swim = torch.cat((pool_x_cnn, pool_x_img_swim), 0)

        # batch_512 = torch.zeros(len(x_img_512), dtype=torch.long).to(device)
        # pool_x_img_512, att_img_512 = self.mpool_img_512(x_img_512, batch_512)
        # att_3.append(att_img_512)
        # pool_x_512 = torch.cat((pool_x_256, pool_x_img_512), 0)

        x = pool_x_swim
        x = F.normalize(x, dim=1)
        k=0
        multi_x = torch.empty((0)).to(device)
        x_img_cnn = self.lin1_img_cnn(x[k])
        x_img_cnn = self.relu(x_img_cnn)
        x_img_cnn = self.norm_img_cnn(x_img_cnn)
        x_img_cnn = self.dropout(x_img_cnn)
        # x_img = self.lin2_img_128(x_img).unsqueeze(0)
        # multi_x = torch.cat((multi_x, x_img), 0)
        k += 1
        x_img_swim = self.lin1_img_swim(x[k])
        x_img_swim = self.relu(x_img_swim)
        x_img_swim = self.norm_img_swim(x_img_swim)
        x_img_swim = self.dropout(x_img_swim)
        # x_img_256 = self.lin2_img_256(x_img_256).unsqueeze(0)
        # multi_x = torch.cat((multi_x, x_img_256), 0)
        # k += 1
        # x_img_512 = self.lin1_img_512(x[k])
        # x_img_512 = self.relu(x_img_512)
        # x_img_512 = self.norm_img_512(x_img_512)
        # x_img_512 = self.dropout(x_img_512)
        # # x_img_512 = self.lin2_img_512(x_img_512).unsqueeze(0)
        # # multi_x = torch.cat((multi_x, x_img_512), 0)
        # # one_x = torch.mean(multi_x, dim=0)



        ##all feature
        # batch = torch.zeros(len(pool_x_img_512), dtype=torch.long).to(device)
        # pool_x_img, att_img= self.mpool_img_512(pool_x_img_512, batch)
        concatenated_feature = torch.cat((x_img_cnn, x_img_swim), dim=0).unsqueeze(0)
        out = nn.Sigmoid()(self.fc(concatenated_feature))
        # out = torch.softmax(out, dim=1)

        # feature = torch.cat((x_img_128, x_img_256, x_img_512),dim=0)
        # x = torch.relu(self.fc1(feature))
        # out = self.fc2(x)
        # out =out.unsqueeze(0)





        # att_3 = []
        # pool_x = torch.empty((0)).to(device)
        #
        # # if 'img' in data_type:
        # batch_128 = torch.zeros(len(x_img_128), dtype=torch.long).to(device)
        # pool_x_img_128, att_img_128 = self.mpool_img_128(x_img_128, batch_128)
        # att_3.append(att_img_128)
        # pool_x = torch.cat((pool_x, pool_x_img_128), 0)
        # # if 'rna' in data_type:
        # batch_256 = torch.zeros(len(x_img_256), dtype=torch.long).to(device)
        # pool_x_img_256, att_img_256 = self.mpool_img_256(x_img_256, batch_256)
        # att_3.append(att_img_256)
        # pool_x = torch.cat((pool_x, pool_x_img_256), 0)
        # # if 'cli' in data_type:
        # batch_512 = torch.zeros(len(x_img_512), dtype=torch.long).to(device)
        # pool_x_img_512, att_img_512 = self.mpool_img_512(x_img_512, batch_512)
        # att_3.append(att_img_512)
        # pool_x = torch.cat((pool_x, pool_x_img_512), 0)
        #
        # x = pool_x
        #
        # x = F.normalize(x, dim=1)
        # fea = x
        #
        # k = 0
        # if 'img' in data_type:
        #     fea_dict['img'] = fea[k]
        #     k += 1
        # if 'rna' in data_type:
        #     fea_dict['rna'] = fea[k]
        #     k += 1
        # if 'cli' in data_type:
        #     fea_dict['cli'] = fea[k]
        #     k += 1
        #
        # k = 0
        # multi_x = torch.empty((0)).to(device)
        #
        # if 'img' in data_type:
        #     x_img = self.lin1_img(x[k])
        #     x_img = self.relu(x_img)
        #     x_img = self.norm_img(x_img)
        #     x_img = self.dropout(x_img)
        #
        #     x_img = self.lin2_img(x_img).unsqueeze(0)
        #     multi_x = torch.cat((multi_x, x_img), 0)
        #     k += 1
        # if 'rna' in data_type:
        #     x_rna = self.lin1_rna(x[k])
        #     x_rna = self.relu(x_rna)
        #     x_rna = self.norm_rna(x_rna)
        #     x_rna = self.dropout(x_rna)
        #
        #     x_rna = self.lin2_rna(x_rna).unsqueeze(0)
        #     multi_x = torch.cat((multi_x, x_rna), 0)
        #     k += 1
        # if 'cli' in data_type:
        #     x_cli = self.lin1_cli(x[k])
        #     x_cli = self.relu(x_cli)
        #     x_cli = self.norm_cli(x_cli)
        #     x_cli = self.dropout(x_cli)
        #
        #     x_cli = self.lin2_rna(x_cli).unsqueeze(0)
        #     multi_x = torch.cat((multi_x, x_cli), 0)
        #     k += 1
        # one_x = torch.mean(multi_x, dim=0)

        return out#(one_x, multi_x), save_fea, (att_2, att_3), fea_dict

# class GCNClassifier(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GCNClassifier, self).__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)
#
#     def forward(self, node_TCGA_image_path_128_fea, node_TCGA_image_path_256_fea, node_TCGA_image_path_512_fea, edge_index_image_128, edge_index_image_256, edge_index_image_512):
#         x_128 = F.relu(self.conv1(node_TCGA_image_path_128_fea, node_TCGA_image_path_256_fea))
#         x_128 = F.dropout(x_128, p=0.5, training=self.training)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)

# class GCN(nn.Module):
#     def __init__(self, in_feats, hidden_size, num_classes):
#         super(GCN, self).__init__()
#         self.conv1 = GraphConv(in_feats, hidden_size)
#         self.conv2 = GraphConv(hidden_size, num_classes)
#
#     def forward(self, g, features):
#         x = F.relu(self.conv1(g, features))
#         x = self.conv2(g, x)
#         return x


class fusion_model_graph(nn.Module):
    def __init__(self, args, in_feats_cnn, in_feats_swim, n_hidden, out_classes, dropout=0.3, num_classes=1):
        super(fusion_model_graph, self).__init__()

        self.img_gnn_cnn = SAGEConv(in_channels=in_feats_cnn, out_channels=out_classes)#args, 2, 1024
        # self.img_gnn_cnn = GATConv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_cnn = GCNConv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_cnn = GENConv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_cnn = GINNet(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_cnn = GPSNet(in_channels=in_feats, out_channels=out_classes, conv=None)
        # self.img_gnn_cnn = GATv2Conv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_cnn = GraphARMAConv(input_dim=in_feats, hidden_dim=256, output_dim=out_classes, num_stacks=2, num_layers=2)
        # self.img_gnn_cnn = PreModel(args, 2, 1024)
        self.img_relu_cnn = GNN_relu_Block(out_classes)

        self.img_gnn_swim = SAGEConv(in_channels=in_feats_swim, out_channels=out_classes)  # args, 2, 1024
        # self.img_gnn_swim = GATConv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_swim = GCNConv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_swim = GENConv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_swim = GINNet(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_swim = GPSNet(in_channels=in_feats, out_channels=out_classes, conv=None)
        # self.img_gnn_swim = GATv2Conv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_swim = GraphARMAConv(input_dim=in_feats, hidden_dim=256, output_dim=out_classes, num_stacks=2, num_layers=2)
        # self.img_gnn_swim = PreModel(args, 2, 1024)
        self.img_relu_swim = GNN_relu_Block(out_classes)

        #TransformerConv

        att_net_img_cnn = nn.Sequential(nn.Linear(out_classes, out_classes // 2), nn.ReLU(), nn.Linear(out_classes // 2, 1))
        self.mpool_img_cnn = my_GlobalAttention(att_net_img_cnn)

        att_net_img_swim = nn.Sequential(nn.Linear(out_classes, out_classes // 2), nn.ReLU(), nn.Linear(out_classes // 2, 1))
        self.mpool_img_swim = my_GlobalAttention(att_net_img_swim)

        # att_net_img_512 = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(), nn.Linear(out_classes // 4, 1))
        # self.mpool_img_512 = my_GlobalAttention(att_net_img_512)



        # self.mae = PretrainVisionTransformer(encoder_embed_dim=out_classes, decoder_num_classes=out_classes, decoder_embed_dim=out_classes, encoder_depth=1, decoder_depth=1, train_type_num=train_type_num)
        # self.mae = ConvNeXtV2(depths=[2, 2, 6, 6], dims=[512, 512, 512, 512])

        # self.mix = MixerBlock(train_type_num, out_classes)

        self.lin1_img_cnn = torch.nn.Linear(out_classes, out_classes // 2)
        self.lin1_img_swim = torch.nn.Linear(out_classes, out_classes // 2)
        # self.lin1_img_512 = torch.nn.Linear(out_classes, out_classes // 4)
        self.lin2_img_cnn = torch.nn.Linear(out_classes // 2, 4)
        self.lin2_img_swim = torch.nn.Linear(out_classes // 2, 4)
        # self.lin2_img_512 = torch.nn.Linear(out_classes // 4, 4)

        # self.lin2_rna = torch.nn.Linear(out_classes // 4, 1)
        # self.lin1_cli = torch.nn.Linear(out_classes, out_classes // 4)
        # self.lin2_cli = torch.nn.Linear(out_classes // 4, 1)

        self.norm_img_cnn = LayerNorm(out_classes // 2)
        self.norm_img_swim = LayerNorm(out_classes // 2)
        # self.norm_img_512 = LayerNorm(out_classes // 4)
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(out_classes // 2 * 2, num_classes)
        # self.fc1 = nn.Linear(n_hidden, n_hidden)
        # self.fc2 = nn.Linear(n_hidden, 4)
        # self.conv2 = GraphConv(n_hidden, 4)

    def forward(self, node_image_path_cnn_fea, node_image_path_swim_fea, edge_index_image_cnn, edge_index_image_swim):
        att_2 = []
        pool_x = torch.empty((0)).to(device)
        x_img_cnn = self.img_gnn_cnn(node_image_path_cnn_fea, edge_index_image_cnn)#loss_img,
        x_img_cnn = torch.relu(x_img_cnn)#self.img_relu_128(x_img_128)
        batch_cnn = torch.zeros(len(x_img_cnn), dtype=torch.long).to(device)
        pool_x_img_cnn, att_img_cnn = self.mpool_img_cnn(x_img_cnn, batch_cnn)
        att_2.append(att_img_cnn)
        pool_x_img_cnn = torch.cat((pool_x, pool_x_img_cnn), 0)

        x_img_swim = self.img_gnn_swim(node_image_path_swim_fea, edge_index_image_swim)  # loss_img,
        x_img_swim = self.img_relu_swim(x_img_swim)
        batch_swim = torch.zeros(len(x_img_swim), dtype=torch.long).to(device)
        pool_x_img_swim, att_img_swim = self.mpool_img_swim(x_img_swim, batch_swim)
        att_2.append(att_img_swim)
        pool_x_img_swim = torch.cat((pool_x_img_cnn, pool_x_img_swim), 0)


        ##跳跃连接
        x_img_cnn = x_img_cnn + pool_x_img_swim[0]
        x_img_swim = x_img_swim + pool_x_img_swim[1]
        # x_img_512 = x_img_512 + pool_x_img_512[2]

        att_3 = []
        pool_x = torch.empty((0)).to(device)
        batch_cnn = torch.zeros(len(x_img_cnn), dtype=torch.long).to(device)
        pool_x_img_cnn, att_img_cnn = self.mpool_img_cnn(x_img_cnn, batch_cnn)
        att_3.append(att_img_cnn)
        pool_x_cnn = torch.cat((pool_x, pool_x_img_cnn), 0)

        batch_swim = torch.zeros(len(x_img_swim), dtype=torch.long).to(device)
        pool_x_img_swim, att_img_swim = self.mpool_img_swim(x_img_swim, batch_swim)
        att_3.append(att_img_swim)
        pool_x_swim = torch.cat((pool_x_cnn, pool_x_img_swim), 0)


        x = pool_x_swim #pool_x_img_swim#
        x = F.normalize(x, dim=1)
        k=0
        # multi_x = torch.empty((0)).to(device)
        x_img_cnn = self.lin1_img_cnn(x[k])
        x_img_cnn = self.relu(x_img_cnn)
        x_img_cnn = self.dropout(x_img_cnn)
        x_img_cnn = self.norm_img_cnn(x_img_cnn)

        # x_img = self.lin2_img_128(x_img).unsqueeze(0)
        # multi_x = torch.cat((multi_x, x_img), 0)
        k += 1
        x_img_swim = self.lin1_img_swim(x[k])
        x_img_swim = self.relu(x_img_swim)
        x_img_swim = self.dropout(x_img_swim)
        x_img_swim = self.norm_img_swim(x_img_swim)

        # x_img_256 = self.lin2_img_256(x_img_256).unsqueeze(0)
        # multi_x = torch.cat((multi_x, x_img_256), 0)





        concatenated_feature = torch.cat((x_img_cnn, x_img_swim), dim=0).unsqueeze(0)
        out = nn.Sigmoid()(self.fc(concatenated_feature))


        return out


