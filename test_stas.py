# from sklearn.metrics import confusion_matrix, precision_score, f1_score,average_precision_score,precision_recall_curve
# from sklearn.metrics import accuracy_score, recall_score
import joblib
import os
from utils.utils import get_optimizer, get_scheduler
import torch
import logging
# from tensorboardX import SummaryWriter
import yaml
import argparse
from pathlib import Path
from collections import OrderedDict
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from GNN import GCN

import os

# 设置环境变量 CUDA_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 你的其他代码...

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
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
    classes = ['N', 'P']

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
    plt.ylabel('True label', fontsize=14)
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

    config_data = config.datasets
    data_cnn_swim = joblib.load('./test_stas.pkl')
    test_data = joblib.load('./stas_patients.pkl')
    model = GCN(node_features=1024, input_size=768, n_hidden=500, output_size=1)#.to(device)
    model2 = model#.to(device)

    # model_swim.head = nn.Identity()
    td = torch.load(r'./GNNbest_model_2.pth', map_location=torch.device('cpu'))
    model2.load_state_dict(td, strict=False)

    ###相关参数
    criterion = torch.nn.BCEWithLogitsLoss()

    print('Test model on test data!')
    fpr = validation_step(model2, test_data, data_cnn_swim, criterion, config)
    if fpr[0]==1:
        label = 'Positive'
    elif fpr[0]==0:
        label = 'Negative'
    else:
        label = 'I donot know!'
    print(label)
    # 打开一个文件以写入模式（'w'）
    with open("./results.txt", "w") as file:
        # 将结果逐行写入文件
        for result in label:
            file.write(result)






def validation_step(model, test_data, data_cnn_swim, criterion, config):
    total_loss = 0
    total_preds = []
    total_labels = []
    total_probs=[]
    total_samples = 0
    ids =[]
    model.eval()
    best_valid_accuracy = 0.0  # 用于保存最佳模型的验证集准确率
    best_model_state_dict = None  # 用于保存最佳模型的参数

    iter = 0
    global loss
    with torch.no_grad():
        for batch, id in enumerate(test_data):
            if 'N' in id.split('.')[0]:
                label = 0
            elif 'P' in id.split('.')[0]:
                label = 1
            else:
                continue  # 如果文件名既不包含N也不包含P，则跳过
            label = torch.Tensor([int(label)]).to(device)
            # label = torch.tensor(label)# label = torch.Tensor([int(label)]).to(device)
            label = label.unsqueeze(1)
            node_image_path_cnn_fea = torch.Tensor(data_cnn_swim[id.split('.')[0]].x_img_cnn).to(device)
            node_image_path_swim_fea = torch.Tensor(data_cnn_swim[id.split('.')[0]].x_img_swim).to(device)
            edge_index_image_cnn = (data_cnn_swim[id.split('.')[0]].x_img_cnn_edge).to(device)
            edge_index_image_swim = (data_cnn_swim[id.split('.')[0]].x_img_swim_edge).to(device)
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
            ids.append(id)

    # 计算各种指标
    # accuracy = accuracy_score(total_labels, total_preds)
    # precision1 = precision_score(total_labels, total_preds, average='macro')
    # recall1 = recall_score(total_labels, total_preds, average='macro')
    # f1 = f1_score(total_labels, total_preds, average='macro')


    return total_preds




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


