import sys

from models import Teacher, GATTeacher, GCN2,LinearClassifier, MLPClassifier
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch
from tqdm import tqdm
import numpy as np
from torch_geometric.datasets import Planetoid, CitationFull
import yaml
from yaml import SafeLoader
import argparse
from os import path as osp
from torch_geometric.utils import remove_isolated_nodes, contains_isolated_nodes, homophily, k_hop_subgraph
import torch_geometric.transforms as T
from torch_geometric.data import Data

def train_edge(data, label, train_mask):
    edge_train_mask = []
    edge_val_mask = []
    edge_test_mask = []
    data_edge = Data()
    entropy_train_mask = []
    for i in range(data.edge_index.shape[-1]):
        if data.train_mask[data.edge_index[0][i]] and  data.train_mask[data.edge_index[1][i]]:
            entropy_train_mask.append(True)
        else:
            entropy_train_mask.append(False)
    real_train_mask = []
    for i in range(data.edge_index.shape[-1]):
        if data.train_mask[data.edge_index[0][i]] and  data.train_mask[data.edge_index[1][i]]:
            real_train_mask.append(True)
        else:
            real_train_mask.append(False)
    real_train_mask = torch.tensor(real_train_mask)
    for i in range(data.edge_index.shape[-1]):
        if train_mask[data.edge_index[0][i]] and  train_mask[data.edge_index[1][i]]:
            edge_train_mask.append(True)
            edge_val_mask.append(False)
            edge_test_mask.append(False)
        elif data.val_mask[data.edge_index[0][i]] and  data.val_mask[data.edge_index[1][i]]:
            edge_train_mask.append(False)
            edge_val_mask.append(True)
            edge_test_mask.append(False)
        elif data.test_mask[data.edge_index[0][i]] and  data.test_mask[data.edge_index[1][i]]:
            edge_train_mask.append(False)
            edge_val_mask.append(False)
            edge_test_mask.append(True)
        else:
            edge_train_mask.append(False)
            edge_val_mask.append(False)
            edge_test_mask.append(False)
    y = []
    for i in range(data.edge_index.shape[-1]):
        if data.y[data.edge_index[0][i]] == data.y[data.edge_index[1][i]]:
            y.append(1)
        else:
            y.append(0)
    train_y = []
    for i in range(data.edge_index.shape[-1]):
        if label[data.edge_index[0][i]] == label[data.edge_index[1][i]]:
            train_y.append(1)
        else:
            train_y.append(0)

    data_edge.train_mask = torch.tensor(edge_train_mask)
    gt_tm = data_edge.train_mask.clone()
    data_edge.val_mask = torch.tensor(edge_val_mask)
    data_edge.test_mask = torch.tensor(edge_test_mask)
    data_edge.y = torch.tensor(y).float()
    data_edge.train_y = torch.tensor(train_y).float()
    data_edge.train_edge_index = data.edge_index.clone()
    #
    # temp = torch.where(data_edge.train_y[data_edge.train_mask] == 0)[0]
    # neg_edge_index = data.edge_index[:,data_edge.train_mask][:,temp]

    # sampled_edge_index = oversampling(data.edge_index, neg_edge_index, label, train_mask, 1)
    #
    #
    #
    #
    #
    # data_edge.train_edge_index = torch.cat((data_edge.train_edge_index, sampled_edge_index), dim = -1)
    #
    # data_edge.train_mask = torch.cat((data_edge.train_mask, torch.ones(sampled_edge_index.shape[-1]) == 1), dim = -1)
    # data_edge.train_y = torch.cat((data_edge.train_y, torch.zeros(sampled_edge_index.shape[-1])), dim = -1)
    gt = data_edge.y.cpu()
    # data_edge.y = torch.cat((data_edge.y, torch.zeros(sampled_edge_index.shape[-1])), dim = -1)
    edge_model = Teacher(in_dim, 128,num_classes).cuda()
    edge_classifier = MLPClassifier(2 * 128,64, 1).cuda()
    data_edge = data_edge.cuda()
    Loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        list(edge_model.parameters()) + list(edge_classifier.parameters()), lr=0.0005, weight_decay=0)
    best_loss = np.inf
    inc = 0
    maj_mask = torch.where(data_edge.y[data_edge.train_mask] == 1)[0]
    min_mask = torch.where(data_edge.y[data_edge.train_mask] == 0)[0]
    # print('maj, min', maj_mask.shape, min_mask.shape)
    # maj_mask = torch.where(data_edge.train_y[data_edge.train_mask] == 1)[0]

    # min_mask = torch.where(data_edge.train_y[data_edge.train_mask] == 0)[0]
    print('maj, min', maj_mask.shape, min_mask.shape)
    for e in tqdm(range(1500)):
        edge_model.train()
        edge_classifier.train()
        optimizer.zero_grad()
        logits = edge_model.embed(data.x,data.edge_index)
        logits1 = logits[data_edge.train_edge_index[0]]
        logits2 = logits[data_edge.train_edge_index[1]]
        logits = torch.cat((logits1, logits2), dim = -1)
        logits = edge_classifier(F.relu(logits)).squeeze(-1)
        if maj_mask.shape[0] > min_mask.shape[0]:
            mask1 = torch.multinomial(maj_mask*1.0, 340, replacement=False)
        else:
            mask1 = torch.multinomial(maj_mask*1.0, 340, replacement=False)
        mask = torch.cat((mask1, min_mask), dim = -1)
        # mask = torch.cat((maj_mask, min_mask), dim = -1)
        loss = Loss(logits[data_edge.train_mask][mask], data_edge.y[data_edge.train_mask][mask])
        loss.backward()
        optimizer.step()
    print(mask1.shape, min_mask.shape)
    edge_model.eval()
    edge_classifier.eval()
    with torch.no_grad():
        logits = edge_model.embed(data.x, data.edge_index)
        logits1 = logits[data.edge_index[0]]
        logits2 = logits[data.edge_index[1]]
        logits = torch.cat((logits1, logits2), dim=-1).detach()
        logits = edge_classifier(F.relu(logits)).sigmoid().squeeze(-1).cpu()
        entropys = (-logits*torch.log(logits)) + (-(1 - logits)*torch.log((1 - logits)))
        entropys[entropy_train_mask] = np.inf
        min_k = torch.topk(-entropys, k= 2000)

    print(accuracy_score(gt, (logits >= 0.5)))
    print((logits[[min_k[1]]] < 0.5).sum())
    mask = torch.ones(data.edge_index.shape[-1])
    zeros = torch.where(logits[min_k[1]] < 0.5)[0]
    print(data_edge.y[min_k[1]][zeros].sum())
    sys.exit()
    mask[min_k[1]][zeros] = 0
    print(zeros.shape, mask[min_k[1]][zeros].shape)
    zeros = torch.where(data_edge.train_y[:real_train_mask.shape[0]][real_train_mask] == 0)[0]
    mask[real_train_mask][zeros] = 0
    print(mask.sum())
    data.edge_index = data.edge_index[:,mask.long()==1]
    print(homophily(data.edge_index, data.y))
    print(data.edge_index.shape)

    return edge_model, edge_classifier


def oversampling(edge_index, neg_edge_index, y, train_mask, sampling_rate):
    center_node_idx = neg_edge_index[0]
    sampled_edge_index = [[],[]]
    for node, start in tqdm(zip(center_node_idx, neg_edge_index[1])):
        node = node.item()
        start = start.item()
        subset = k_hop_subgraph(node,1, edge_index)[0]
        class_start = y[start]
        for sampled_node in subset:
            if sampled_node in train_mask:
                if y[sampled_node] == class_start:
                    continue
                if np.random.random() > (1-sampling_rate):
                    sampled_edge_index[0] += [class_start,sampled_node]
                    sampled_edge_index[1] += [sampled_node, class_start]
            elif np.random.random() > (1-sampling_rate):
                sampled_edge_index[0] += [class_start,sampled_node]
                sampled_edge_index[1] += [sampled_node, class_start]
    return torch.tensor(sampled_edge_index, device='cuda:0')


def train(data, model, label, train_mask,first = True, round = None, model_name = None):
    if label is None and train_mask is None:
        label_tmp = data.y[data.train_mask]
        train_mask = data.train_mask
    else:
        label_tmp = label[train_mask]
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay= 5e-4)
    if model_name == 'gcn2':
        reg_params = list(model.convs.parameters())
        non_reg_params = list(model.fcs.parameters())
        if first:
            optimizer = torch.optim.Adam([
                dict(params=reg_params, weight_decay= 1e-2 - 0.9995e-2*round/20),
                dict(params=non_reg_params, weight_decay=5e-4 - 4.95e-4*round/20)
            ], lr= 0.01 - 0.0095*round/20)
        else:
            optimizer = torch.optim.Adam([
                dict(params=reg_params, weight_decay= 5e-6),
                dict(params=non_reg_params, weight_decay=5e-6)
            ], lr=0.001)
    best_val_loss = np.inf
    inc = 0
    for e in tqdm(range(1500)):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[train_mask], label_tmp)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.eval()
            logits = model(data.x, data.edge_index).detach()
            loss = criterion(logits[data.val_mask], data.y[data.val_mask])
            if loss < best_val_loss:
                best_val_loss = loss
                best_epoch = e
                torch.save({'model_state_dict': model.state_dict()},f'checkpoints/best_teacher_{model_name}_model.pt')
                inc = 0
            else:
                inc += 1
            if inc > 100:
                break
    print(f'loading: {best_epoch}')



def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']
    name = 'dblp' if name == 'DBLP' else name

    return (CitationFull if name == 'dblp' else Planetoid)(
        path,
        name,
        split = 'public',
        transform = T.NormalizeFeatures()
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)
    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    learning_rate = config['learning_rate']
    hid_dim = config['num_hidden']
    proj_hid_dim = config['num_proj_hidden']
    proj_dim = config['num_proj']
    num_layers = config['num_layers']
    dropout = config['dropout']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']
    patience = 20

    path = osp.join('datasets', args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]

    num_classes = dataset.num_classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    in_dim = dataset.num_features
    new_edge_index = [[],[]]
    print(data.edge_index.shape)
    teacher_accs = []
    student_accs = []
    for _ in range(10):
        # teacher_model = GATTeacher(in_dim, 512, num_classes).cuda()
        teacher_model = GCN2(model_level = 'Node', dim_node = in_dim, dim_hidden = 64, num_classes = num_classes, alpha = 0.1, theta = 0.5, num_layers = 64,
                                     shared_weights = False, dropout = 0.6).cuda()
        # model = GATTeacher(in_dim, 512, num_classes).cuda()
        label = data.y.clone()
        train_mask = data.train_mask.clone()
        back_label = data.y.clone()
        back_train_mask = data.train_mask.clone()
        round = 1
        augmented = 0
        max_round = int(data.x.shape[0]/140) + 1
        best_val_loss = np.inf
        criterion = torch.nn.CrossEntropyLoss()
    #     while True:
    #         if round >= max_round:
    #             break
    #         print(round)
    #         round += 1
    #     #run teacher model
    #         if round < 5:
    #             train(data, teacher_model, back_label, back_train_mask, True, round,'gcn2')
    #         else:
    #             train(data, teacher_model, back_label, back_train_mask, False, round,'gcn2')
    #
    #         teacher_model.eval()
    #         with torch.no_grad():
    #             teacher_model.load_state_dict(torch.load('checkpoints/best_teacher_gcn2_model.pt')['model_state_dict'])
    #             logits = teacher_model(data.x, data.edge_index).detach()
    #             logits = F.softmax(logits, dim=-1)
    #             y_pred = logits[data.test_mask].argmax(-1).cpu()
    #             y_test = data.y[data.test_mask].cpu()
    #             logits = teacher_model(data.x, data.edge_index).detach()
    #             loss = criterion(logits[data.val_mask], data.y[data.val_mask])
    #             print('val_loss:',loss.item())
    #             #     if augmented:
    #             #         if loss < best_val_loss:
    #             #             best_val_loss = loss
    #             #             torch.save({'model_state_dict': model.state_dict()},'checkpoints/best_model.pt')
    #             teacher_acc = accuracy_score(y_test, y_pred)
    #             print(teacher_acc)
    #             #     # sys.exit()
    #             outputs = teacher_model(data.x, data.edge_index)
    #             outputs = F.softmax(outputs, dim = -1)
    #             entropys = (-outputs*torch.log(outputs)).sum(-1)
    #             entropys[train_mask] = np.inf
    #             if data.y.shape[0] - train_mask.sum() < 140:
    #                 topk = data.y.shape[0] - train_mask.sum()
    #             else:
    #                 topk = 140
    #             min_k = torch.topk(-entropys, k= topk)
    #             train_mask[min_k[1]] = True
    #             print('teacher pseudo label')
    #             print(data.y[min_k[1]].eq(outputs.argmax(-1)[min_k[1]]).sum(-1)/min_k[1].shape[0])
    #             label[min_k[1]] = outputs.argmax(-1)[min_k[1]]
    #
    #
    #
    #
    #         if round < 5:
    #             train(data, model, label, train_mask, True, round,'gat')
    #         else:
    #             train(data, model, label, train_mask, False, round,'gat')
    #
    #         model.eval()
    #         with torch.no_grad():
    #             model.load_state_dict(torch.load('checkpoints/best_teacher_gat_model.pt')['model_state_dict'])
    #             logits = model(data.x, data.edge_index).detach()
    #             logits = F.softmax(logits, dim=-1)
    #             y_pred = logits[data.test_mask].argmax(-1).cpu()
    #             y_test = data.y[data.test_mask].cpu()
    #             logits = model(data.x, data.edge_index).detach()
    #             loss = criterion(logits[data.val_mask], data.y[data.val_mask])
    #             print('val_loss:',loss.item())
    #         #     if augmented:
    #         #         if loss < best_val_loss:
    #         #             best_val_loss = loss
    #         #             torch.save({'model_state_dict': model.state_dict()},'checkpoints/best_model.pt')
    #             student_acc = accuracy_score(y_test, y_pred)
    #             print(student_acc)
    #         #     # sys.exit()
    #             outputs = model(data.x, data.edge_index)
    #             outputs = F.softmax(outputs, dim = -1)
    #             entropys = (-outputs*torch.log(outputs)).sum(-1)
    #             entropys[back_train_mask] = np.inf
    #             if data.y.shape[0] - back_train_mask.sum() < 140:
    #                 topk = data.y.shape[0] - back_train_mask.sum()
    #             else:
    #                 topk = 140
    #             min_k = torch.topk(-entropys, k= topk)
    #             back_train_mask[min_k[1]] = True
    #             print('student pseudo label')
    #             print(data.y[min_k[1]].eq(outputs.argmax(-1)[min_k[1]]).sum(-1)/min_k[1].shape[0])
    #             back_label[min_k[1]] = outputs.argmax(-1)[min_k[1]]
    #     teacher_accs.append(teacher_acc)
    #     student_accs.append(student_acc)
    # print(np.mean(teacher_accs), np.std(teacher_accs), np.mean(student_accs), np.std(student_accs))
        # if augmented:
        #     break
        # if  round == 10:
        #     edge_model, edge_classifier = train_edge(data, label, train_mask)
        #     # sys.exit()
        #     # model = GATTeacher(in_dim, 512, num_classes).cuda()
        #     augmented = 1
        #     model = Teacher(in_dim, 1024, num_classes).cuda()
        #     label = data.y.clone()
        #     train_mask = data.train_mask.clone()
        #     round = 1
    # model.eval()
    # with torch.no_grad():
    #     model.load_state_dict(torch.load('checkpoints/best_model.pt')['model_state_dict'])
    #     logits = model(data.x, data.edge_index).detach()
    #     logits = F.softmax(logits, dim=-1)
    #     y_pred = logits[data.test_mask].argmax(-1).cpu()
    #     y_test = data.y[data.test_mask].cpu()
    #     print('Final result:',accuracy_score(y_test, y_pred))
        train(data, teacher_model, back_label, back_train_mask, True, 0,'gcn2')
        teacher_model.eval()
        with torch.no_grad():
            teacher_model.load_state_dict(torch.load('checkpoints/best_teacher_gcn2_model.pt')['model_state_dict'])
            logits = teacher_model(data.x, data.edge_index).detach()
            logits = F.softmax(logits, dim=-1)
            y_pred = logits[data.test_mask].argmax(-1).cpu()
            y_test = data.y[data.test_mask].cpu()
            teacher_acc = accuracy_score(y_test, y_pred)
            print(teacher_acc)
            teacher_accs.append(teacher_acc)
    print(np.mean(teacher_accs), np.std(teacher_accs))