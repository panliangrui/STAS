import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
# from your_module import SemanticAttention  # 请替换成实际的 SemanticAttention 模块


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=1024):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1024, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        out_emb = (beta * z)#.sum(1)  # (N, D * K)
        att_mp = beta.mean(0).squeeze()

        return out_emb, att_mp
class HANLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HANLayer, self).__init__()
        # self.gat_layers = nn.ModuleList([
        #     GATConv(in_channels, out_channels),
        #     GATConv(in_channels, out_channels)
        # ])
        self.gat_layers1 = GATConv(in_channels, out_channels)
        self.gat_layers2 = GATConv(in_channels, out_channels)

        self.semantic_attention = SemanticAttention(out_channels, out_channels)
        self.activation = nn.PReLU()

    def forward(self, x, edge_index):
        # 在第一个GATConv层上进行前向传播
        h1 = self.gat_layers1(x, edge_index)
        h1 = self.activation(h1)

        # 在第二个GATConv层上进行前向传播
        h2 = self.gat_layers2(x, edge_index)
        h2 = self.activation(h2)

        # 使用语义注意力层进行注意力权重计算
        # h = []
        # h = h.append(h1)
        # h = h.append(h2)
        h = h1+h2

        # semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        out, att_weights = self.semantic_attention(h)

        # 使用注意力权重对两个分支的结果进行加权融合
        h = att_weights * h1 + (1 - att_weights) * h2

        return h

class HAN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(HAN, self).__init__()
        self.han_layers = HANLayer(in_channels, out_channels)#nn.ModuleList([HANLayer(in_channels, out_channels) for _ in range(num_layers)])
        self.activation = nn.PReLU()

    def forward(self, x, edge_index):
        # 初始化输入特征
        h = x

        h_all = []
        # 逐层应用HANLayer
        # for layer in self.han_layers:
        #     # 在每个HANLayer中进行前向传播
        #     h_all1 = layer(h, edge_index)
        #     h_all.append(h_all1)
        h_all = self.han_layers(h, edge_index)
        # 使用注意力权重对两个分支的结果进行加权融合
        h_ = h_all#[0] + h_all[1]

        # 应用激活函数
        h_1 = self.activation(h_)

        # 返回最终的节点表示
        return h_1

class PreModel(nn.Module):
    def __init__(self, num_metapath):
        super(PreModel, self).__init__()
        self.encoder = HAN(in_channels=1024, out_channels=512, num_layers=num_metapath)
        self.decoder = HAN(in_channels=512, out_channels=512, num_layers=1)
        self.attr_restoration_loss = nn.MSELoss()
        self.encoder_to_decoder = nn.Linear(512, 512, bias=False)
        self.mp_edge_recon_loss = nn.MSELoss()
        self.encoder_to_decoder_edge_recon = nn.Linear(512, 512, bias=False)
        self.mp2vec_feat_pred_loss = nn.MSELoss()
        self.enc_out_to_mp2vec_feat_mapping = nn.Sequential(
            nn.Linear(512, 512, bias=True),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 512, bias=True),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 512, bias=True)
        )

    def forward(self, x, edge_index):
        # 编码器的前向传播
        encoder_output = self.encoder(x, edge_index)

        # 解码器的前向传播
        decoder_output = self.decoder(encoder_output, edge_index)

        # 属性恢复损失
        attr_loss = self.attr_restoration_loss(decoder_output, x)

        # 编码器到解码器的映射
        encoder_to_decoder_output = self.encoder_to_decoder(encoder_output)

        # 边重建损失
        edge_recon_loss = self.mp_edge_recon_loss(encoder_to_decoder_output, x)

        # 编码器到解码器的边重建映射
        encoder_to_decoder_edge_recon_output = self.encoder_to_decoder_edge_recon(encoder_to_decoder_output)

        # mp2vec特征预测损失
        mp2vec_feat_pred_loss = self.mp2vec_feat_pred_loss(encoder_to_decoder_edge_recon_output, x)

        # enc_out到mp2vec_feat_mapping的前向传播
        mp2vec_feat_mapping_output = self.enc_out_to_mp2vec_feat_mapping(encoder_output)

        # 返回损失和预测结果
        return {
            'attr_loss': attr_loss,
            'edge_recon_loss': edge_recon_loss,
            'mp2vec_feat_pred_loss': mp2vec_feat_pred_loss,
            'mp2vec_feat_mapping_output': mp2vec_feat_mapping_output
        }

# 创建模型实例
num_metapath = 2  # 设置 num_metapath 的值
model = PreModel(num_metapath)

# 将模型传递给优化器并进行训练等操作
