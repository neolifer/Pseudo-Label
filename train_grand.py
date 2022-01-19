from __future__ import division
from __future__ import print_function
from models import Teacher, GATTeacher, GCN2,LinearClassifier, MLPClassifier
from sklearn.metrics import accuracy_score
import time
import argparse
import numpy as np
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid, CitationFull
from pygcn.utils import load_data, accuracy
from pygcn.models import GCN, MLP
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch_geometric.transforms as T
from torch_geometric.utils import dense_to_sparse,from_scipy_sparse_matrix
from torch_geometric.data import Data
from os import path as osp

scaler = StandardScaler()
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=1024,
                    help='Number of hidden units.')
parser.add_argument('--input_droprate', type=float, default=0.5,
                    help='Dropout rate of the input layer (1 - keep probability).')
parser.add_argument('--hidden_droprate', type=float, default=0.5,
                    help='Dropout rate of the hidden layer (1 - keep probability).')
parser.add_argument('--dropnode_rate', type=float, default=0.5,
                    help='Dropnode rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--order', type=int, default=5, help='Propagation step')
parser.add_argument('--sample', type=int, default=4, help='Sampling times of dropnode')
parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1., help='Lamda')
parser.add_argument('--dataset', type=str, default='cora', help='Data set')
parser.add_argument('--cuda_device', type=int, default=4, help='Cuda device')
parser.add_argument('--use_bn', action='store_true', default=False, help='Using Batch Normalization')

args = parser.parse_args()


def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']
    name = 'dblp' if name == 'DBLP' else name

    return (CitationFull if name == 'dblp' else Planetoid)(
        path,
        name,
        split = 'public',
        transform = T.NormalizeFeatures()
    )



def propagate(feature, A, order):
    #feature = F.dropout(feature, args.dropout, training=training)
    x = feature
    y = feature
    for i in range(order):
        x = torch.spmm(A, x).detach_()
        #print(y.add_(x))
        y.add_(x)

    return y.div_(order+1.0).detach_()

def rand_prop(features, training):
    n = features.shape[0]
    drop_rate = args.dropnode_rate
    drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)

    if training:

        masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)

        features = masks.cuda() * features



    else:
        #     pass
        features = features * (1. - drop_rate)
    features = propagate(features, A, args.order)
    return features

def consis_loss(logps, temp=args.tem):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p/len(ps)
    #p2 = torch.exp(logp2)

    sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p-sharp_p).pow(2).sum(1))
    loss = loss/len(ps)
    return args.lam * loss

def train_grand(optimizer,epoch, grand_model, back_label, idx_train):
    t = time.time()

    X = features

    grand_model.train()
    optimizer.zero_grad()
    X_list = []
    K = args.sample
    for k in range(K):
        X_list.append(rand_prop(X, training=True))

    output_list = []
    for k in range(K):
        output_list.append(torch.log_softmax(grand_model(X_list[k]), dim=-1))


    loss_train = 0.
    for k in range(K):
        loss_train += F.nll_loss(output_list[k][idx_train], back_label[idx_train])


    loss_train = loss_train/K
    #loss_train = F.nll_loss(output_1[idx_train], labels[idx_train]) + F.nll_loss(output_1[idx_train], labels[idx_train])
    #loss_js = js_loss(output_1[idx_unlabel], output_2[idx_unlabel])
    #loss_en = entropy_loss(output_1[idx_unlabel]) + entropy_loss(output_2[idx_unlabel])
    loss_consis = consis_loss(output_list)

    loss_train = loss_train + loss_consis
    acc_train = accuracy(output_list[0][idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        grand_model.eval()
        X = rand_prop(X,training=False)
        output = grand_model(X)
        output = torch.log_softmax(output, dim=-1)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    # print('Epoch: {:04d}'.format(epoch+1),
    #       'loss_train: {:.4f}'.format(loss_train.item()),
    #       'acc_train: {:.4f}'.format(acc_train.item()),
    #       'loss_val: {:.4f}'.format(loss_val.item()),
    #       'acc_val: {:.4f}'.format(acc_val.item()),
    #       'time: {:.4f}s'.format(time.time() - t))
    return loss_val.item(), acc_val.item()
def Train_grand(grand_model, back_label, idx_train):
    # Train model
    t_total = time.time()
    loss_values = []
    acc_values = []
    bad_counter = 0
    # best = args.epochs + 1
    loss_best = np.inf
    acc_best = 0.0

    loss_mn = np.inf
    acc_mx = 0.0

    best_epoch = 0
    optimizer = optim.Adam(grand_model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    if round > 2:
        optimizer = optim.Adam(grand_model.parameters(),
                               lr= 0.001 - 0.0009*round/max_round, weight_decay= 1e-4)
        args.dropnode_rate = 0.2 - 0.1*round/max_round
    for epoch in tqdm(range(args.epochs)):
        # if epoch < 200:
        #   l, a = train(epoch, True)
        #   loss_values.append(l)
        #   acc_values.append(a)
        #   continue

        l, a = train_grand(optimizer,epoch, grand_model, back_label, idx_train)
        loss_values.append(l)
        acc_values.append(a)

        # print(bad_counter)
        #
        if loss_values[-1] <= loss_mn or acc_values[-1] >= acc_mx:# or epoch < 400:
            if loss_values[-1] <= loss_best: #and acc_values[-1] >= acc_best:
                loss_best = loss_values[-1]
                acc_best = acc_values[-1]
                best_epoch = epoch
                torch.save(grand_model.state_dict(), args.dataset +'.pkl')

            loss_mn = np.min((loss_values[-1], loss_mn))
            acc_mx = np.max((acc_values[-1], acc_mx))
            bad_counter = 0
        else:
            bad_counter += 1
        if bad_counter > 100:
            break
        # print(bad_counter, loss_mn, acc_mx, loss_best, acc_best, best_epoch)
        # if bad_counter == args.patience:
        #     print('Early stop! Min loss: ', loss_mn, ', Max accuracy: ', acc_mx)
        #     print('Early stop model validation loss: ', loss_best, ', accuracy: ', acc_best)
        #     break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    grand_model.load_state_dict(torch.load(args.dataset +'.pkl'))



def test_grand(grand_model):
    grand_model.eval()
    X = features
    X = rand_prop(X, training=False)
    output = grand_model(X)
    output = torch.log_softmax(output, dim=-1)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()

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


if __name__ == '__main__':

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.cuda.set_device(args.cuda_device)
    dataset = args.dataset
    accs_teacher = []
    accs_student = []
    for _ in range(10):
        path = osp.join('datasets', args.dataset)
        dataset = get_dataset(path, args.dataset)
        data = dataset[0]
        num_classes = dataset.num_classes
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)
        in_dim = dataset.num_features
        A, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset.lower())

        if args.cuda:
            features = features.cuda()
            A = A.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()
            train_labels = labels.clone()
        args.dropnode_rate = 0.5
        grand_model = MLP(nfeat= in_dim,
                          nhid=args.hidden,
                          nclass= num_classes,
                          input_droprate=args.input_droprate,
                          hidden_droprate=args.hidden_droprate,
                          use_bn = args.use_bn).cuda()
        # model = GATTeacher(in_dim, 512, num_classes,dropout=0.6).cuda()
        model = MLP(nfeat= in_dim,
                    nhid=args.hidden,
                    nclass= num_classes,
                    input_droprate=args.input_droprate,
                    hidden_droprate=args.hidden_droprate,
                    use_bn = args.use_bn).cuda()
        label = data.y.clone()
        train_mask = data.train_mask.clone()
        back_label = data.y.clone()
        back_train_mask = data.train_mask.clone()
        round = 1
        augmented = 0
        max_round = int(data.x.shape[0]/140) + 1
        best_val_loss = np.inf
        criterion = torch.nn.CrossEntropyLoss()
        idx_train = torch.where(data.train_mask)[0]
        idx_val = torch.where(data.val_mask)[0]
        idx_test = torch.where(data.test_mask)[0]
        labels = data.y
        # while True:
        #     idx_train = torch.where(back_train_mask)[0]
        #     if round >= max_round:
        #         break
        #     print(round)
        #     round += 1
        #     Train_grand()
        #     acc_teacher = test_grand()
        #     grand_model.eval()
        #     X = features
        #     X = rand_prop(X, training=False)
        #     outputs = grand_model(X)
        #     outputs = F.softmax(outputs, dim = -1)
        #     entropys = (-outputs*torch.log(outputs)).sum(-1)
        #     entropys[train_mask] = np.inf
        #     if data.y.shape[0] - train_mask.sum() < 140:
        #         topk = data.y.shape[0] - train_mask.sum()
        #     else:
        #         topk = 140
        #     min_k = torch.topk(-entropys, k= topk)
        #     train_mask[min_k[1]] = True
        #     print('teacher pseudo label')
        #     print(data.y[min_k[1]].eq(outputs.argmax(-1)[min_k[1]]).sum(-1)/min_k[1].shape[0])
        #     label[min_k[1]] = outputs.argmax(-1)[min_k[1]]
        #
        #
        #     if round < 5:
        #         train(data, model, label, train_mask, True, round,'gat')
        #     else:
        #         train(data, model, label, train_mask, False, round,'gat')
        #
        #     model.eval()
        #     with torch.no_grad():
        #         model.load_state_dict(torch.load('checkpoints/best_teacher_gat_model.pt')['model_state_dict'])
        #         logits = model(data.x, data.edge_index).detach()
        #         logits = F.softmax(logits, dim=-1)
        #         y_pred = logits[data.test_mask].argmax(-1).cpu()
        #         y_test = data.y[data.test_mask].cpu()
        #         logits = model(data.x, data.edge_index).detach()
        #         loss = criterion(logits[data.val_mask], data.y[data.val_mask])
        #         print('val_loss:',loss.item())
        #         #     if augmented:
        #         #         if loss < best_val_loss:
        #         #             best_val_loss = loss
        #         #             torch.save({'model_state_dict': model.state_dict()},'checkpoints/best_model.pt')
        #         student_acc = accuracy_score(y_test, y_pred)
        #         print(accuracy_score(y_test, y_pred))
        #         #     # sys.exit()
        #         outputs = model(data.x, data.edge_index)
        #         outputs = F.softmax(outputs, dim = -1)
        #         entropys = (-outputs*torch.log(outputs)).sum(-1)
        #         entropys[back_train_mask] = np.inf
        #         if data.y.shape[0] - back_train_mask.sum() < 140:
        #             topk = data.y.shape[0] - back_train_mask.sum()
        #         else:
        #             topk = 140
        #         min_k = torch.topk(-entropys, k= topk)
        #         back_train_mask[min_k[1]] = True
        #         print('student pseudo label')
        #         print(data.y[min_k[1]].eq(outputs.argmax(-1)[min_k[1]]).sum(-1)/min_k[1].shape[0])
        #         back_label[min_k[1]] = outputs.argmax(-1)[min_k[1]]
        while True:
            idx_train = torch.where(back_train_mask)[0]
            if round >= max_round:
                break
            print(round)
            round += 1
            Train_grand(grand_model,back_label,idx_train)
            acc_teacher = test_grand(grand_model)
            grand_model.eval()
            X = features
            X = rand_prop(X, training=False)
            outputs = grand_model(X)
            outputs = F.softmax(outputs, dim = -1)
            entropys = (-outputs*torch.log(outputs)).sum(-1)
            entropys[train_mask] = np.inf
            if data.y.shape[0] - train_mask.sum() < 140:
                topk = data.y.shape[0] - train_mask.sum()
            else:
                topk = 140
            min_k = torch.topk(-entropys, k= topk)
            train_mask[min_k[1]] = True
            print('teacher pseudo label')
            print(data.y[min_k[1]].eq(outputs.argmax(-1)[min_k[1]]).sum(-1)/min_k[1].shape[0])
            label[min_k[1]] = outputs.argmax(-1)[min_k[1]]


            idx_train = torch.where(train_mask)[0]
            Train_grand(model,label, idx_train)
            acc_student = test_grand(model)
            model.eval()
            X = features
            X = rand_prop(X, training=False)
            outputs = model(X)
            outputs = F.softmax(outputs, dim = -1)
            entropys = (-outputs*torch.log(outputs)).sum(-1)
            entropys[back_train_mask] = np.inf
            if data.y.shape[0] - back_train_mask.sum() < 140:
                topk = data.y.shape[0] - back_train_mask.sum()
            else:
                topk = 140
            min_k = torch.topk(-entropys, k= topk)
            back_train_mask[min_k[1]] = True
            print('student pseudo label')
            print(data.y[min_k[1]].eq(outputs.argmax(-1)[min_k[1]]).sum(-1)/min_k[1].shape[0])
            back_label[min_k[1]] = outputs.argmax(-1)[min_k[1]]
        accs_teacher.append(acc_teacher)
        accs_student.append(acc_student)
    print(accs_teacher, accs_student)
    print(np.mean(accs_teacher), np.std(accs_teacher), np.mean(accs_student), np.std(accs_student))
