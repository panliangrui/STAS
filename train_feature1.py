# from data.fold5 import ordered_yaml
from sklearn.metrics import confusion_matrix, precision_score, f1_score
from data.dataloader import CancerDataset
from sklearn.metrics import accuracy_score, recall_score
import joblib
import argparse
import os
from pathlib import Path
from utils import get_loss, get_model, get_optimizer, get_scheduler
import torch
import yaml
from tqdm import tqdm
import logging
from models.GSAGE import GraphSAGE
from tensorboardX import SummaryWriter
from models import multiview
import yaml
from models.transformer import Transformer
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from models.SGFormer import SGFormer
from collections import OrderedDict
from torch_geometric.data import Data
from torch_geometric.explain import GNNExplainer
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from models.GNN import GCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('-config', type=str, help='./configs', default="")
parser.add_argument('-seed', type=int, help='random seed of the run', default=611)
parser.add_argument('-fold', default=1, type=int, help='random seed of the run')
parser.add_argument('-model_path', default='./results/models')
parser.add_argument('-save_dir', default='../results')
parser.add_argument('-stop_criterion', default='auroc')
parser.add_argument('-num_epochs', default=50, type=int)
parser.add_argument('-val_check_interval', default=2, type=int)
parser.add_argument("--n_hidden", type=int, default=512, help="Model middle dimension")
parser.add_argument("--drop_out_ratio", type=float, default=0.2, help="Drop_out_ratio")
parser.add_argument("--out_classes", type=int, default=512, help="Model out dimension")
parser.add_argument("--train_use_type", type=list, default=['128', '256', '512'],  #
                    help='train_use_type,Please keep the relative order of img, rna, cli')

parser.add_argument("--num_classes", type=int, default=1, help="Model out dimension")
parser.add_argument("--adjust_lr_ratio", type=float, default=0.5, help="adjust_lr_ratio")
parser.add_argument("--if_adjust_lr", action='store_true', default=True, help="if_adjust_lr")

args = parser.parse_args()

opt_path = args.config
default_config_path = "COAD_config.yml"

CONFIG_DIR = Path("./config")

if opt_path == "":
    opt_path = CONFIG_DIR / default_config_path




def configure_optimizers(config):
    optimizer = get_optimizer(
        name=config.optimizer,
        model=config.model,
        lr=config.lr,
        wd=config.wd,
    )
    if config.lr_scheduler:
        scheduler = get_scheduler(
            config.lr_scheduler,
            optimizer,
            **config.lr_scheduler_config,
        )
        return [optimizer], [scheduler]
    else:
        return [optimizer]


logs = set()


def init_log_save(save_dir, name, level=logging.INFO):
    save_name = save_dir / 'logger.txt'
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.FileHandler(save_name)
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def adjust_learning_rate(optimizer, lr, epoch, lr_step=20, lr_gamma=0.5):
    lr = lr * (lr_gamma ** (epoch // lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


import numpy as np


def calculate_specificity(total_preds, total_labels, num_classes):
    # 将预测和标签转换为numpy数组
    total_preds = np.array(total_preds)
    total_labels = np.array(total_labels)

    # 初始化特异性列表
    specificities = []

    # 计算每个类别的特异性
    for class_index in range(num_classes):
        true_negatives = np.sum((total_preds != class_index) & (total_labels != class_index))
        false_positives = np.sum((total_preds == class_index) & (total_labels != class_index))
        specificity = true_negatives / (true_negatives + false_positives)
        specificities.append(specificity)

    return specificities


from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np


def calculate_multiclass_auc_roc(total_preds, total_labels, num_classes):
    # 将预测和标签转换为numpy数组
    total_preds = np.array(total_preds)
    total_labels = np.array(total_labels)

    # 初始化AUC和ROC曲线字典
    auc_scores = {}
    roc_curves = {}

    # 计算每个类别的AUC和ROC曲线
    for class_index in range(num_classes):
        # 将当前类别作为正例，其他类别作为负例
        y_true = (total_labels == class_index).astype(int)
        y_score = total_preds  # 在这里，我们不再使用索引，因为total_preds是一维数组

        # 计算AUC
        # auc_scores[class_index] = roc_auc_score(y_true, y_score)
        try:
            auc_scores[class_index] = roc_auc_score(y_true, y_score)  ## y_true=ground_truth
        except ValueError:
            pass

        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_curves[class_index] = (fpr, tpr)

    return auc_scores, roc_curves

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def get_confusion_matrix(total_labels, total_preds):
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(total_labels, total_preds)

    # 定义类别标签
    classes = [0, 1, 2, 3]

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # 添加标签
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()




def ordered_yaml():
    """
    yaml orderedDict support
    """
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def main():
    opt_path = "./config.yml"
    with open(opt_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
        print(f"Loaded configs from {opt_path}")
        config = argparse.Namespace(**config)

    base_path = Path(config.save_dir)
    config.logging_name = f'{config.name}_{config.model}_{"-".join(config.cohorts)}_{config.norm}_{config.target}' if config.name != 'debug' else 'debug'
    # base_path = base_path / config.logging_name
    base_path.mkdir(parents=True, exist_ok=True)
    model_path = base_path / 'models'/ 'subtype'
    model_path.mkdir(parents=True, exist_ok=True)
    result_path = base_path / 'Graph'/ 'results'
    result_path.mkdir(parents=True, exist_ok=True)

    # 保存训练loss的文档
    logger = init_log_save(result_path, 'global', logging.INFO)
    logger.propagate = 0
    tb = SummaryWriter(log_dir=result_path)

    config_data = config.datasets
    config_data = argparse.Namespace(**config_data)
    # path_256_cnn = joblib.load(config_data.path_256_cnn)
    # data_cnn = joblib.load('./data/cnn_1024.pkl')

    # data_val = joblib.load('./data/val150s.pkl')
    data_train = joblib.load('./data/all_data.pkl')
    # data_train_val = joblib.load('./data/cnn_swim_1024_768_512_all_new.pkl')

    # train_data = []

    # train_data = list(data_train_val.keys())
    #
    # with open('./traindata.txt', 'w') as file:
    #     for file_name in train_data:
    #         file.write(file_name + '\n')

    # train_data = []
    # with open("./data/train.txt", "r") as file:
    #     # 逐行读取文件内容
    #     for line in file:
    #         # 假设每行的数据用空格分隔，并且 diamante 在第一个位置
    #         diamante = line.split()[0]
    #         train_data.append(diamante)
    # #
    # val_data = []
    # with open("./data/val.txt", "r") as file:
    #     # 逐行读取文件内容
    #     for line in file:
    #         # 假设每行的数据用空格分隔，并且 diamante 在第一个位置
    #         diamante = line.split()[0]
    #         val_data.append(diamante)
    #切片选择800和200
    # train_data = list(data_train.keys())[:800]
    # val_data = list(data_train.keys())[-200:]
    # 随机选择800和200
    import random
    all_keys = list(data_train.keys())
    # test_data = random.sample(all_keys1, 200)
    # all_keys = [key for key in all_keys1 if key not in test_data]
    # with open('./data/test200.txt', 'w') as train_file:
    #     for file_name in test_data:
    #         train_file.write(file_name + '\n')

    from sklearn.metrics import accuracy_score, precision_score, f1_score
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True)
    k = 0
    for train_data_i, val_data_i in kf.split(all_keys):
        train_data = [all_keys[i] for i in train_data_i]
        val_data = [all_keys[i] for i in val_data_i]
        with open('./data/kanval5fold{:}.txt'.format(k), 'w') as val_file:
            for file_name in val_data:
                val_file.write(file_name + '\n')
    # for k in range(args.fold):

        # model = multiview.fusion_model_graph(args, in_feats_cnn=100, in_feats_swim=100, n_hidden=50, out_classes=50,
        #                                    dropout=args.drop_out_ratio, num_classes=args.num_classes).to(
        #     device)
        # model = GCN(node_features=1024, input_size=768, n_hidden=500, output_size=1).to(device)
        from models.kan_GCN import kanGCN
        model = kanGCN(node_features=1024, input_size=768, n_hidden=500, output_size=1).to(device)

        # model = SGFormer(in_channels=1024, node_fe= 768, hidden_channels=500, out_channels=1).to(device)


        # model = Transformer(num_classes=1, input_dim=1024, input_dim1=768).to(device)


        #model = GraphSAGE(nodefea=1024, in_channels=768, hidden_channels=500, num_layers=2).to(device)

        ###相关参数
        import torch.optim as optim
        # optimizer = optim.Adam(model.parameters(), lr=0.001,  eps=1e-8, betas=(0.9, 0.999), weight_decay=0.001)##get_optimizer(name=config.optimizer['opt_method'], model=model, lr=config.optimizer['lr'], wd=config.optimizer['weight_decay'])#
        optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)
        criterion = torch.nn.BCEWithLogitsLoss()
        # criterion = torch.nn.CrossEntropyLoss()
        # lr = config.lr

        #####训练模型
        training_range = tqdm(range(100), nrows=3) #config.train['num_epochs']

        print(f"Start training model")
        # best_valid_accuracy =0.0
        for epoch in training_range:

            model.train()

            t_loss, total_samples, total_preds, total_labels, total_probs = training_step(model, train_data, data_train, optimizer, criterion, config)

            # 计算各种指标
            accuracy = accuracy_score(total_labels, total_preds)
            precision = precision_score(total_labels, total_preds, average='macro')
            recall = recall_score(total_labels, total_preds, average='macro')
            f1 = f1_score(total_labels, total_preds, average='macro')
            # 计算特异性的平均值
            # 计算混淆矩阵
            conf_matrix = confusion_matrix(total_labels, total_preds)
            # 计算特异度
            specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

            # 计算AUC,ROC
            auc = roc_auc_score(total_labels, total_probs)
            tb.add_scalar('train_loss_total', t_loss / total_samples, epoch)
            logger.info('epoch: {:}, Train loss: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, '
                        'f1: {:.3f}, specificity: {:.3f}, AUC: {:.3f}'.format(epoch, t_loss / total_samples,
                                                                 accuracy, precision, recall, f1, specificity, auc))
            # 打印每个 epoch 的损失值
            # print(
            #     f"Epoch [{epoch + 1}/{training_range}], Train Loss: {t_loss / total_samples},accuracy: {accuracy},precision: {precision},recall: {recall}, f1: {f1}, specificity: {specificity},AUC: {auc}")

            # if accuracy > best_valid_accuracy:
            #     best_valid_accuracy = accuracy
            #     best_model_state_dict = model.state_dict()
            #
            #     # 保存在验证集上表现最佳的模型参数
            #     # current_model_path = os.path.join(model_path, f'best_model_{k}.pth')
            #     current_model_path = os.path.join('./results/model_graph', f'SGFbest_model_{k}.pth')
            #     torch.save(best_model_state_dict, current_model_path)
            if epoch % 5 == 0:
                print('Validation model on validation data!')
                evaluate_result = validation_step(model, val_data, data_train, criterion, epoch, config, tb, model_path,logger, k)
                print(evaluate_result)
            # if epoch == (100-1):
            #     print('Test model on test data!')
            #     test_result = test_step(model, test_data, data_cnn, data_swim, epoch, tb, result_path)
        k=k+1
        # explainer = GNNExplainer(model, epochs=8, return_type='log_prob')
        # label = train_data[0]
        # node_image_path_cnn_fea = torch.Tensor(train_data[label].x_img_cnn).to(device)
        # node_image_path_swim_fea = torch.Tensor(train_data[label].x_img_swim).to(device)
        # edge_index_image_cnn = (train_data[label].x_img_cnn_edge).to(device)
        # edge_index_image_swim = (train_data[label].x_img_swim_edge).to(device)
        # #node_image_path_cnn_fea_mask, node_image_path_swim_fea_mask, edge_index_image_cnn_mask, edge_index_image_swim_mask \
        # node_feat_mask, edge_mask= explainer.explain_graph(node_image_path_cnn_fea, node_image_path_swim_fea, edge_index_image_cnn, edge_index_image_swim)
        # ax, G = explainer.visualize_subgraph(-1, node_feat_mask, edge_mask, None)
        # plt.show()


def training_step(model, train_data, data_cnn_swim, optimizer, criterion, config):

    total_loss = 0
    total_preds = []
    total_probs=[]
    total_labels = []
    total_correct = 0
    total_samples = 0
    # all_data_cnn = {}

    for batch, id in enumerate(train_data):
        # cnn_fea_1, swim_fea_1 = data_cnn_swim[id].to(device), data_cnn_swim,[id].to(device)#, label  # .to(device)
        # 遍历文件夹中的所有文件
        # for file_name in os.listdir(self.label_path):
        if 'N' in id:
            label = 0
        elif 'P' in id:
            label = 1
        else:
            continue  # 如果文件名既不包含N也不包含P，则跳过
        label = torch.Tensor([int(label)]).to(device)
        # label = torch.tensor(label)# label = torch.Tensor([int(label)]).to(device)
        label = label.unsqueeze(1)
        optimizer.zero_grad()
        node_image_path_cnn_fea=torch.Tensor(data_cnn_swim[id].x_img_cnn).to(device)
        node_image_path_swim_fea=torch.Tensor(data_cnn_swim[id].x_img_swim).to(device)
        edge_index_image_cnn= (data_cnn_swim[id].x_img_cnn_edge).to(device)
        edge_index_image_swim =(data_cnn_swim[id].x_img_swim_edge).to(device)
        logits = model(node_image_path_cnn_fea, node_image_path_swim_fea, edge_index_image_cnn, edge_index_image_swim) #model(node_image_path_cnn_fea.unsqueeze(0), node_image_path_swim_fea.unsqueeze(0))#
        # fea1.append(x3)
        # fea2.append(x5)
        # fea3.append(concatenated_feature)
        # ids.append(id)
        # data_all = Data(xfea1024=(x3.cpu().detach()).numpy(), fea768 = (x5.cpu().detach()).numpy(), all= (concatenated_feature.cpu().detach()).numpy())
        # all_data_cnn[id] = data_all


        if config.task == "binary":
            loss = criterion(logits, label)
        else:
            loss = criterion(logits, label)

        loss.backward()
        optimizer.step()
        # 计算准确率和召回率
        logits = torch.sigmoid(logits)
        preds = np.where(logits.cpu().detach().numpy()[0][0]>= 0.5, 1, 0)  # 假设输出是 logits，使用阈值 0 将其转换为二元预测
        # correct = (preds == label).sum().item()
        total_samples += len(label)

        # 累加正确预测的数量和样本数量
        # total_correct += correct
        total_preds.append(preds.item())
        total_probs.append(logits.cpu().detach().numpy()[0][0])
        total_labels.append(int(label.item()))
        total_samples += len(label)
        total_loss += loss
    # joblib.dump(all_data_cnn, './feature_heatmap1.pkl')


    return total_loss, total_samples, total_preds, total_labels,total_probs  # , total_correct, total_tp, total_tn, total_fp, total_fn


def validation_step(model, val_data, data_val, criterion, epoch, config, tb, model_path, logger,k):
    total_loss = 0
    total_preds = []
    total_labels = []
    total_probs=[]
    total_samples = 0
    model.eval()
    best_AUC = 0.0  # 用于保存最佳模型的验证集准确率
    best_model_state_dict = None  # 用于保存最佳模型的参数

    iter = 0
    global loss
    with torch.no_grad():
        for batch, id in enumerate(val_data):
            # cnn_fea_1, swim_fea_1 = cnn_fea[id].to(device), swim_fea[id].to(device)  # , label  # .to(device)
            # 遍历文件夹中的所有文件
            # for file_name in os.listdir(self.label_path):
            if 'N' in id:
                label = 0
            elif 'P' in id:
                label = 1
            else:
                continue  # 如果文件名既不包含N也不包含P，则跳过
            label = torch.Tensor([int(label)]).to(device)
            # label = torch.tensor(label)# label = torch.Tensor([int(label)]).to(device)
            label = label.unsqueeze(1)
            node_image_path_cnn_fea = torch.Tensor(data_val[id].x_img_cnn).to(device)
            node_image_path_swim_fea = torch.Tensor(data_val[id].x_img_swim).to(device)
            edge_index_image_cnn = (data_val[id].x_img_cnn_edge).to(device)
            edge_index_image_swim = (data_val[id].x_img_swim_edge).to(device)
            logits = model(node_image_path_cnn_fea, node_image_path_swim_fea, edge_index_image_cnn,
                           edge_index_image_swim)

            if config.task == "binary":
                loss = criterion(logits, label)
            else:
                loss = criterion(logits, label)

            # 计算准确率和召回率
            logits = torch.sigmoid(logits)
            preds = np.where(logits.cpu().detach().numpy()[0][0]>= 0.5, 1, 0)  # 假设输出是 logits，使用阈值 0 将其转换为二元预测
            # correct = (preds == label).sum().item()
            total_samples += len(label)

            # 累加正确预测的数量和样本数量
            # total_correct += correct
            total_preds.append(preds.item())
            total_probs.append(logits.cpu().detach().numpy()[0][0])
            total_labels.append(int(label.item()))
            total_samples += len(label)
            total_loss += loss

            # print(f"iter [{batch + 1}], Loss: {loss}")
            # iter += 1

    # 计算各种指标
    accuracy = accuracy_score(total_labels, total_preds)
    precision = precision_score(total_labels, total_preds, average='macro')
    recall = recall_score(total_labels, total_preds, average='macro')
    f1 = f1_score(total_labels, total_preds, average='macro')
    # 计算特异性的平均值
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(total_labels, total_preds)
    # 计算特异度
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

    # 计算AUC,ROC
    auc = roc_auc_score(total_labels, total_probs)

    print("Confusion Matrix:\n", conf_matrix)


    tb.add_scalar('val_loss_total', total_loss / total_samples, epoch)
    logger.info('epoch: {:}, Val loss: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, '
                    'f1: {:.3f}, specificity: {:.3f}, AUC: {:.3f}'.format(epoch, total_loss / total_samples,
                                                             accuracy, precision, recall, f1, specificity, auc))
    # 打印每个 epoch 的损失值
    print(f"Epoch [{epoch + 1}], Val Loss: {total_loss / total_samples},accuracy: {accuracy},precision: {precision},recall: {recall}, f1: {f1}, specificity: {specificity},AUC: {auc}")
    ### 如果当前模型的验证集准确率优于之前的最佳准确率，则保存当前模型参数
    if auc > best_AUC:
        best_AUC = auc
        best_model_state_dict = model.state_dict()

        # 保存在验证集上表现最佳的模型参数
        # current_model_path = os.path.join(model_path, f'best_model_{k}.pth')
        current_model_path = os.path.join('./results/models', f'kanGCN_RMSprop_best_model_{k}.pth')
        torch.save(best_model_state_dict, current_model_path)

    return "Val finish: {:.3f},".format(epoch)


# def test_step(model, test_data, cnn_fea, swim_fea, epoch, tb, result_path):
#     total_loss = 0
#     total_preds = []
#     total_labels = []
#     total_probs=[]
#     total_samples = 0
#     model.eval()
#
#     global loss
#     with torch.no_grad():
#         for batch, id in enumerate(test_data):
#             cnn_fea_1, swim_fea_1 = cnn_fea[id].to(device), swim_fea[id].to(device)  # , label  # .to(device)
#             # 遍历文件夹中的所有文件
#             # for file_name in os.listdir(self.label_path):
#             if 'N' in id:
#                 label = 0
#             elif 'P' in id:
#                 label = 1
#             else:
#                 continue  # 如果文件名既不包含N也不包含P，则跳过
#             label = torch.Tensor([int(label)]).to(device)
#             # label = torch.tensor(label)# label = torch.Tensor([int(label)]).to(device)
#             label = label.unsqueeze(1)
#             node_image_path_cnn_fea = torch.Tensor(cnn_fea_1.x_img_cnn).to(device)
#             node_image_path_swim_fea = torch.Tensor(swim_fea_1.x_img_swim).to(device)
#             edge_index_image_cnn = (cnn_fea_1.x_img_cnn_edge).to(device)
#             edge_index_image_swim = (swim_fea_1.x_img_swim_edge).to(device)
#             logits = model(node_image_path_cnn_fea, node_image_path_swim_fea, edge_index_image_cnn,
#                            edge_index_image_swim)
#
#             # 计算准确率和召回率
#             logits = torch.sigmoid(logits)
#             preds = np.where(logits.cpu().detach().numpy()[0][0]>= 0.5, 1, 0)   # 假设输出是 logits，使用阈值 0 将其转换为二元预测
#             # correct = (preds == label).sum().item()
#             total_samples += len(label)
#
#             # 累加正确预测的数量和样本数量
#             # total_correct += correct
#             total_preds.append(preds.item())
#             total_probs.append(logits.cpu().detach().numpy()[0][0])
#             total_labels.append(int(label.item()))
#             total_samples += len(label)
#             total_loss += loss
#
#     # 计算各种指标
#     accuracy = accuracy_score(total_labels, total_preds)
#     precision = precision_score(total_labels, total_preds, average='macro')
#     recall = recall_score(total_labels, total_preds, average='macro')
#     f1 = f1_score(total_labels, total_preds, average='macro')
#     # 计算特异性的平均值
#     # 计算混淆矩阵
#     conf_matrix = confusion_matrix(total_labels, total_preds)
#     # 计算特异度
#     specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
#
#     # 计算AUC,ROC
#     auc = roc_auc_score(total_labels, total_probs)
#     # 计算 ROC 曲线
#     fpr, tpr, thresholds = roc_curve(total_labels, total_probs)
#
#     # 绘制 ROC 曲线
#     plt.plot(fpr, tpr, label='ROC Curve (AUC = %0.2f)' % auc)
#     plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Guessing')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend()
#     plt.show()
#     # 计算混淆矩阵
#     # conf_matrix = confusion_matrix(y_true, y_pred)
#
#     print("Confusion Matrix:\n", conf_matrix)
#
#     # 打印每个 epoch 的损失值
#     print(
#         f"Epoch [{epoch + 1}], Val Loss: {total_loss / total_samples},accuracy: {accuracy},precision: {precision},recall: {recall}, f1: {f1}, specificity: {specificity},AUC: {auc}")
#
#     return 0




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


