import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, nodefea=1024, in_channels=768, hidden_channels=500, num_layers=2):
        super(GraphSAGE, self).__init__()
        self.convs = ModuleList()
        self.convs.append(SAGEConv(nodefea, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.lin = nn.Linear(hidden_channels, 1)


        self.convs1 = ModuleList()
        self.convs1.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs1.append(SAGEConv(hidden_channels, hidden_channels))
        self.lin1 = nn.Linear(hidden_channels, 1)

    def forward(self, x, x1, edge_index, edge_index1):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.lin(x)

        for conv in self.convs1:
            x1 = F.relu(conv(x1, edge_index1))
        x1 = self.lin(x1)

        x = torch.squeeze(x)
        x1 = torch.squeeze(x1)  # .mean(0)

        concatenated_feature  = x1.unsqueeze(0)#torch.cat((x, x1), dim=0).unsqueeze(0)  # concatenated_feature = x1.unsqueeze(0)#
        # self.fc = nn.Linear(len(concatenated_feature), 1).to(device)
        # x = self.fc(concatenated_feature)
        out = torch.mean(concatenated_feature)  # nn.Sigmoid()(torch.mean(concatenated_feature))
        out = torch.unsqueeze(out, 0).unsqueeze(1)


        return out

# # 示例用法
# # 定义模型参数
# num_nodes = 1000  # 图中节点数
# num_features = 16  # 每个节点的特征维度
# num_classes = 2  # 二分类任务
#
# # 生成随机节点特征和边索引
# x = torch.randn(num_nodes, num_features)
# edge_index = torch.randint(0, num_nodes, (2, 5000))  # 随机生成5000条边
#
# # 构建GraphSAGE模型
# model = GraphSAGE(num_features, hidden_channels=64, num_layers=2)
#
# # 定义损失函数和优化器
# criterion = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
# # 模型训练
# for epoch in range(100):
#     optimizer.zero_grad()
#     output = model(x, edge_index)
#     y_true = torch.randint(0, 2, (num_nodes,))  # 随机生成节点的真实标签
#     loss = criterion(output.view(-1), y_true.float())
#     loss.backward()
#     optimizer.step()
#     print('Epoch {}, Loss: {:.4f}'.format(epoch+1, loss.item()))
