#单独为每个文件构图

import os
from torch_geometric.data import Data

import torch
import numpy as np
import joblib
import nmslib
from itertools import chain
import dgl
from scipy.stats import pearsonr
class Hnsw:
    """
    KNN model cloned from https://github.com/mahmoodlab/Patch-GCN/blob/master/WSI-Graph%20Construction.ipynb
    """

    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=True):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        # this is the actual nmslib part, hopefully the syntax should
        # be pretty readable, the documentation also has a more verbiage
        # introduction: https://nmslib.github.io/nmslib/quickstart.html
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params, print_progress=self.print_progress)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn):
        # the knnQuery returns indices and corresponding distance
        # we will throw the distance away for now
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices

knn_model = Hnsw(space='l2')
def get_node(t_img_fea_cnn, t_img_fea_swim):
    f_img_cnn = []
    f_img_swim = []

    for z in t_img_fea_cnn:
        f_img_cnn.append(t_img_fea_cnn[z])

    for z in t_img_fea_swim:
        f_img_swim.append(t_img_fea_swim[z])

    return f_img_cnn, f_img_swim
def perform_pca(data, n):
    # 假设使用PCA对数据进行降维处理
    # 这里只是一个示例，实际应用中需要使用合适的降维方法
    # 在这里，我们使用PCA将数据降维到n维
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n)
    reduced_data = pca.fit_transform(data)
    return reduced_data

folder_path_256_cnn = './stas_all/np'
folder_path_256_swim = './stas_all/np_swim'
# save_path_cnn = 'P:\\xiangya2\\STAS\\stas_all\\graph\\cnn'
# save_path_swim = 'P:\\xiangya2\\STAS\\stas_all\\graph\\swim'
# save_path_all = 'P:\\xiangya2\\STAS\\stas_all\\graph\\cnn_swim'
# path = "P:\\xiangya2\\STAS\\fenlei_nandp\\data"
all_data_cnn = {}
all_data_swim = {}
all_data = {}

# 打开文件
# with open('./val.txt', 'r') as file:
#     # 读取文件的每一行，并提取第一列数据
#     first_column_data = [line.split()[0] for line in file]
#
# print(first_column_data)
first_column_data = joblib.load('./stas_patients.pkl')
# 遍历文件夹中的所有文件
# for file_name in os.listdir(folder_path_256_cnn):
#     if file_name.endswith('.pkl'):
for file_name in first_column_data:
    if '.pkl' not in file_name:
        file_name = file_name +'.pkl'
    image_path_cnn_fea = os.path.join(folder_path_256_cnn, file_name)
    image_path_cnn_fea = joblib.load(image_path_cnn_fea)

    image_path_swim_fea = os.path.join(folder_path_256_swim, file_name)
    image_path_swim_fea = joblib.load(image_path_swim_fea)

    # from data.dataloader import get_node#, get_edge_index_image

    node_image_path_cnn_fea, node_image_path_swim_fea = get_node(image_path_cnn_fea, image_path_swim_fea)
    node_image_path_cnn_fea = torch.Tensor(np.stack(node_image_path_cnn_fea))
    node_image_path_swim_fea = torch.Tensor(np.stack(node_image_path_swim_fea))
    #构建cnn的边
    n_patches = len(image_path_cnn_fea)  # .shape[0]
    # 使用列表推导式提取所有值
    all_values = [value for value in image_path_cnn_fea.values()]

    # 使用vstack函数将所有值堆叠在一起
    image_path_cnn_fea = np.vstack(all_values)
    radius = 9
    # Construct graph using spatial coordinates
    knn_model.fit(image_path_cnn_fea)

    a = np.repeat(range(n_patches), radius - 1)
    b = np.fromiter(
        chain(
            *[knn_model.query(image_path_cnn_fea[v_idx], topn=radius)[1:] for v_idx in range(n_patches)]
        ), dtype=int
    )
    edge_spatial = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)

    # Create edge types
    edge_type = []
    edge_sim = []
    for (idx_a, idx_b) in zip(a, b):
        metric = pearsonr
        corr = metric(image_path_cnn_fea[idx_a], image_path_cnn_fea[idx_b])[0]
        edge_type.append(1 if corr > 0 else 0)
        edge_sim.append(corr)

    # Construct dgl heterogeneous graph
    graph = dgl.graph((edge_spatial[0, :], edge_spatial[1, :]))
    image_path_cnn_fea = torch.tensor(image_path_cnn_fea, device='cpu').float()
    # 获取图中的边
    edges1 = graph.edges()
    edge_index_image_cnn = torch.tensor(torch.stack(edges1, dim=0), dtype=torch.long)

    ##goujianswim的边
    n_patches = len(image_path_swim_fea)  # .shape[0]
    # 使用列表推导式提取所有值
    all_values = [value for value in image_path_swim_fea.values()]

    # 使用vstack函数将所有值堆叠在一起
    image_path_swim_fea = np.vstack(all_values)
    radius = 9
    # Construct graph using spatial coordinates
    knn_model.fit(image_path_swim_fea)

    a = np.repeat(range(n_patches), radius - 1)
    b = np.fromiter(
        chain(
            *[knn_model.query(image_path_swim_fea[v_idx], topn=radius)[1:] for v_idx in range(n_patches)]
        ), dtype=int
    )
    edge_spatial = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)

    # Create edge types
    edge_type = []
    edge_sim = []
    for (idx_a, idx_b) in zip(a, b):
        metric = pearsonr
        corr = metric(image_path_swim_fea[idx_a], image_path_swim_fea[idx_b])[0]
        edge_type.append(1 if corr > 0 else 0)
        edge_sim.append(corr)

    # Construct dgl heterogeneous graph
    graph = dgl.graph((edge_spatial[0, :], edge_spatial[1, :]))
    image_path_swim_fea = torch.tensor(image_path_swim_fea, device='cpu').float()
    # 获取图中的边
    edges1 = graph.edges()
    edge_index_image_swim = torch.tensor(torch.stack(edges1, dim=0), dtype=torch.long)


    # data_swim = Data(x_img_swim=node_image_path_swim_fea, x_img_swim_edge=edge_index_image_swim)#, id=file_name.split('.')[0])
    # data_cnn = Data(x_img_cnn=node_image_path_cnn_fea, x_img_cnn_edge=edge_index_image_cnn)#, id=file_name.split('.')[0])
    data_all = Data(x_img_cnn=node_image_path_cnn_fea, x_img_cnn_edge = edge_index_image_cnn, x_img_swim=node_image_path_swim_fea, x_img_swim_edge = edge_index_image_swim)#, id = file_name)
    # all_data_cnn[file_name.split('.')[0]] = data_cnn
    # all_data_swim[file_name.split('.')[0]] = data_swim

    all_data[file_name.split('.')[0]] = data_all
    # full_path_cnn = '{}/{}'.format(save_path_cnn, file_name)
    # full_path_swim = '{}/{}'.format(save_path_swim, file_name)
    # full_path_all = '{}/{}'.format(save_path_all, file_name)
    # joblib.dump(data_cnn, full_path_cnn)
    # joblib.dump(data_swim, full_path_swim)
    # joblib.dump(data_all, full_path_all)

        # file_path = '{}.pkl'.format(file_name.split('.')[0])
# file_path_cnn = os.path.join(save_path_cnn, '{}.pkl'.format(file_name.split('.')[0]))
# file_path_swim = os.path.join(save_path_swim, '{}.pkl'.format(file_name.split('.')[0]))
# joblib.dump(all_data_cnn, './cnn_1024_test.pkl')
# joblib.dump(all_data_swim, './swim_768_test.pkl')
joblib.dump(all_data, './test_stas.pkl')
print('finish!')



