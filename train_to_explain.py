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


def train(data, model, label, train_mask,first = True, round = None, model_name = None):
    if label is None and train_mask is None:
        label_tmp = data.y[data.train_mask]
        train_mask = data.train_mask
    else:
        label_tmp = label[train_mask]
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005, weight_decay= 1e-3)
    if model_name == 'gcnii':
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
    elif 'gcn' in model_name:
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005, weight_decay= 5e-6)
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
                torch.save({'model_state_dict': model.state_dict()},f'checkpoints/{args.dataset}_best_teacher_{model_name}_{round}_model1.pt')
                inc = 0
            else:
                inc += 1
            if inc > 100:
                break
    print(f'loading: {best_epoch}')



def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed']


    return Planetoid(
        path,
        name,
        split = 'public'
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
    if args.dataset in ['Cora','CiteSeer']:
        pl_per_round = 140
    else:
        pl_per_round = 120
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
        # teacher_model = GCN2(model_level = 'Node', dim_node = in_dim, dim_hidden = 64, num_classes = num_classes, alpha = 0.1, theta = 0.5, num_layers = 64,
        #                      shared_weights = False, dropout = 0.6).cuda()
#         teacher_model = Teacher(in_dim, 512, num_class = num_classes, dropout=0.5).cuda()
        teacher_model = GATTeacher(in_dim, 128, num_classes,dropout=0.6).cuda()
        for conv in teacher_model.conv:
            conv.get_vertex = False
        # model = Teacher(in_dim, 512, num_class = num_classes,dropout=0.5).cuda()
        if args.dataset == 'CiteSeer':
            model = GATTeacher(in_dim, 512, num_classes,dropout=0.7).cuda()
        elif args.dataset == 'Cora':
            model = GATTeacher(in_dim, 512, num_classes,dropout=0.6).cuda()
        if args.dataset == 'PubMed':
            model = GATTeacher(in_dim, 128, num_classes,dropout=0.6).cuda()
        for conv in model.conv:
            conv.get_vertex = False
        # model = teacher_model = Teacher(in_dim, 512, num_class = num_classes).cuda()
        label = data.y.clone()
        train_mask = data.train_mask.clone()
        back_label = data.y.clone()
        back_train_mask = data.train_mask.clone()
        round = 1
        augmented = 0
        max_round = int(data.x.shape[0]/pl_per_round) + 1
        best_val_loss = np.inf
        criterion = torch.nn.CrossEntropyLoss()
        while True:
            if round >= max_round:
                break
            print(round)
            round += 1
        #run teacher model
            if round < 5:
                train(data, teacher_model, back_label, back_train_mask, True, round,'gat1')
            else:
                train(data, teacher_model, back_label, back_train_mask, False, round,'gat1')

            teacher_model.eval()
            with torch.no_grad():
                teacher_model.load_state_dict(torch.load(f'checkpoints/{args.dataset}_best_teacher_gcn1_{round}_model1.pt')['model_state_dict'])
                logits = teacher_model(data.x, data.edge_index).detach()
                logits = F.softmax(logits, dim=-1)
                y_pred = logits[data.test_mask].argmax(-1).cpu()
                y_test = data.y[data.test_mask].cpu()
                # logits = teacher_model(data.x, data.edge_index).detach()
                # loss = criterion(logits[data.val_mask], data.y[data.val_mask])
                # print('val_loss:',loss.item())
                #     if augmented:
                #         if loss < best_val_loss:
                #             best_val_loss = loss
                #             torch.save({'model_state_dict': model.state_dict()},'checkpoints/best_model.pt')
                teacher_acc = accuracy_score(y_test, y_pred)
                print(teacher_acc)
                #     # sys.exit()
                outputs = teacher_model(data.x, data.edge_index).detach()
                outputs = F.softmax(outputs, dim = -1)
                entropys = (-outputs*torch.log(outputs)).sum(-1)
                entropys[train_mask] = np.inf
                if data.y.shape[0] - train_mask.sum() < pl_per_round:
                    topk = data.y.shape[0] - train_mask.sum()
                else:
                    topk = pl_per_round
                min_k = torch.topk(-entropys, k= topk)
                # choice = np.random.choice(min_k[1].cpu(), topk)
                # print(len(choice))
                train_mask[min_k[1]] = True
                print('teacher pseudo label')
                print(data.y[min_k[1]].eq(outputs.argmax(-1)[min_k[1]]).sum(-1)/min_k[1].shape[0])
                label[min_k[1]] = outputs.argmax(-1)[min_k[1]]




            if round < 5:
                train(data, model, label, train_mask, True, round,'gat2')
            else:
                train(data, model, label, train_mask, False, round,'gat2')

            model.eval()
            with torch.no_grad():
                model.load_state_dict(torch.load(f'checkpoints/{args.dataset}_best_teacher_gat2_{round}_model1.pt')['model_state_dict'])
                logits = model(data.x, data.edge_index).detach()
                logits = F.softmax(logits, dim=-1)
                y_pred = logits[data.test_mask].argmax(-1).cpu()
                y_test = data.y[data.test_mask].cpu()
                # logits = model(data.x, data.edge_index).detach()
                # loss = criterion(logits[data.val_mask], data.y[data.val_mask])
                # print('val_loss:',loss.item())
            #     if augmented:
            #         if loss < best_val_loss:
            #             best_val_loss = loss
            #             torch.save({'model_state_dict': model.state_dict()},'checkpoints/best_model.pt')
                student_acc = accuracy_score(y_test, y_pred)
                print(student_acc)
            #     # sys.exit()
                outputs = model(data.x, data.edge_index).detach()
                outputs = F.softmax(outputs, dim = -1)
                entropys = (-outputs*torch.log(outputs)).sum(-1)
                entropys[back_train_mask] = np.inf
                if data.y.shape[0] - back_train_mask.sum() < pl_per_round:
                    topk = data.y.shape[0] - back_train_mask.sum()
                else:
                    topk = pl_per_round
                min_k = torch.topk(-entropys, k= topk)
                # choice = np.random.choice(min_k[1].cpu(), topk)
                # print(len(choice))
                back_train_mask[min_k[1]] = True
                print('student pseudo label')
                print(data.y[min_k[1]].eq(outputs.argmax(-1)[min_k[1]]).sum(-1)/min_k[1].shape[0])
                back_label[min_k[1]] = outputs.argmax(-1)[min_k[1]]
        # sys.exit()
        teacher_accs.append(teacher_acc)
        student_accs.append(student_acc)
    print(np.mean(teacher_accs), np.std(teacher_accs), np.mean(student_accs), np.std(student_accs))
    # print(np.mean(teacher_accs), np.std(teacher_accs))
