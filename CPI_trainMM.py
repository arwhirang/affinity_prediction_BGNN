import sys
import math
import time
import pickle
import numpy as np
from collections import defaultdict
import torch
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from rdkit import Chem

from pdbbind_utilsMM import *
from CPI_modelMM import *

from scipy import linalg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)  # <== not working on cuda 2... only cuda 0?

cudaDevid = 0
max_num_seq = 2000
max_num_atoms = 200


# function for generating batch data
def runTheModel(net, sample, indexForModel, criterion1, isTraining=False):  # indexForModel 0:KIKD, 1:IC50
    vec_l, vec_p, adj_inter, mask_l, mask_p, values = sample# adj_l, adj_p,
    aff_pred, pair_pred = net(vec_l, vec_p, adj_inter, mask_l, mask_p, indexForModel, isTrain=isTraining)# adj_l, adj_p,
    #print(aff_pred)
    assert len(aff_pred) == len(values)
    loss_aff = criterion1(aff_pred, values)
    #loss_pairwise = criterion1(pairwise_pred, tmpFRIMat)#, L_mask, P_mask, pairwise_exist)

    # rak ver
    loss = loss_aff #+ loss_pairwise

    # pairwise_loss += float(loss_pairwise.data * batch_size)
    if isTraining:
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5)
    return aff_pred, pair_pred, float(loss_aff.data * batch_size), 0#float(loss_pairwise.data * batch_size)


# train and evaluate
def train_and_eval(tr_kikd_dl, valid_kikd_dl, tr_ic50_dl, test_dl, vocab_atom_size, batch_size=32, num_epoch=30,
                   loadFromSavedMoel=False, isTrainProcess=True):
    net = Net(vocab_atom_size)
    net.cuda(cudaDevid)

    if not loadFromSavedMoel:
        net.apply(weights_init)
    else:
        net.load_state_dict(torch.load("./current_model.pt"))

    kikdIndex = torch.Tensor([1, 0]).cuda(cudaDevid)
    ic50Index = torch.Tensor([0, 1]).cuda(cudaDevid)

    criterion1 = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0005, weight_decay=0,
                           amsgrad=False)  # True)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 100, 250], gamma=0.9)
    min_rmse = 1000
    best_epoch = 0
    perf_name = ['RMSE', 'Pearson', 'Spearman',]# 'avg pairwise AUC']
    if isTrainProcess:
        for epoch in range(num_epoch):
            total_loss = 0
            affinity_loss = 0
            pairwise_loss = 0

            # kikdLen = math.ceil(len(train_kikd[0]) / batch_size)
            # ic50Len = math.ceil(len(train_ic50[0]) / batch_size)
            # lenTrain = max(kikdLen, ic50Len)
            net.train()
            for i_batch, sample in enumerate(tr_kikd_dl):
                optimizer.zero_grad()
                _, _, affloss, pairloss = runTheModel(net, sample, kikdIndex, criterion1, isTraining=True)
                total_loss += affloss# + pairloss
                affinity_loss += affloss
                pairwise_loss += pairloss
                optimizer.step()
            """
            for i_batch, sample in enumerate(tr_ic50_dl):
                optimizer.zero_grad()
                _, _, affloss, pairloss = runTheModel(net, sample, ic50Index, criterion1, isTraining=True)
                total_loss += affloss# + pairloss
                affinity_loss += affloss
                pairwise_loss += pairloss
                optimizer.step()
            """
            scheduler.step()

            loss_list = [total_loss, affinity_loss , pairwise_loss]
            loss_name = ['total loss', 'affinity loss' , 'pairwise loss']
            print_loss = [loss_name[i] + ' ' + str(loss_list[i] / (len(tr_kikd_dl)+ len(tr_ic50_dl))) for i in
                          range(len(loss_name))]  #
            print('epoch:', epoch, ' '.join(print_loss))

            net.eval()
            with torch.no_grad():
                valid_performance, valid_output = test(net, valid_kikd_dl, kikdIndex, batch_size)
                print_perf = [perf_name[i] + ' ' + str(round(valid_performance[i], 6)) for i in range(len(perf_name))]
                print('valid', len(valid_output), ' '.join(print_perf))

                test_performance, test_output = test(net, test_dl, kikdIndex, batch_size)
                print_perf = [perf_name[i] + ' ' + str(round(test_performance[i], 6)) for i in range(len(perf_name))]
                print('test ', len(test_output), ' '.join(print_perf))

                if valid_performance[0] < min_rmse:
                    min_rmse = valid_performance[0]
                    torch.save(net.state_dict(), "./current_model.pt")
                    best_epoch = epoch
                    print('model saved <== current model is better than min_rmse')

    # test phase
    net.load_state_dict(torch.load("./current_model.pt"))
    print("loaded from the best epoch:", best_epoch)
    net.eval()
    with torch.no_grad():
        test_performance, test_output = test(net, test_dl, kikdIndex, batch_size)
        print_perf = [perf_name[i] + ' ' + str(round(test_performance[i], 6)) for i in range(len(perf_name))]
        print('test ', len(test_output), ' '.join(print_perf))
    print('Finished.')


def test(net, test_data, kikdIndex, batch_size):  # only KIKD
    output_list = []
    label_list = []
    pairwise_auc_list = []
    criterionTest = nn.MSELoss()
    loss_val = 0

    for i_batch, sample in enumerate(test_data):
        _, _, _, mask_l, mask_p, val_values = sample
        aff_pred, pair_pred, affloss, pairloss = runTheModel(net, sample, kikdIndex, criterionTest, isTraining=False)

        output_list += aff_pred.cpu().detach().numpy().reshape(-1).tolist()
        label_list += val_values.reshape(-1).tolist()
        loss_val += affloss + pairloss
        # for j in range(len(val_L_mask)):
        #     num_vertex = min(int(torch.sum(val_L_mask[j, :])), max_num_atoms)
        #     num_residue = min(int(torch.sum(val_P_mask[j, :])), max_num_seq)
        #
        #     pairwise_pred_j = pairwise_pred[j, :num_vertex, :num_residue].cpu().detach().numpy().reshape(-1)
        #     pairwise_label_j = val_tmpFRIMat[j, :num_vertex, :num_residue].cpu().reshape(-1)
        #     pairwise_auc_list.append(roc_auc_score(pairwise_label_j, pairwise_pred_j))

    output_list = np.array(output_list)
    label_list = np.array(label_list)

    print("validation loss:", str(loss_val / len(test_data)))
    rmse_value, pearson_value, spearman_value = reg_scores(label_list, output_list)
    # average_pairwise_auc = np.mean(pairwise_auc_list)
    test_performance = [rmse_value, pearson_value, spearman_value,] #average_pairwise_auc]
    return test_performance, output_list


class MolDataset(Dataset):
    def __init__(self, datalist):
        self.datalist = datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        vec_latoms, adj_inter, vec_patoms, values, pdbid = self.datalist[idx]
        sample = {
            'vec_latoms': vec_latoms,
            'vec_patoms': vec_patoms,
            'adj_inter': adj_inter,
            'values': values,
            'pdbid': pdbid,
        }
        return sample


def collate_fn(batch):  # collate means collect and combine
    vec_l = np.zeros((len(batch), max_num_atoms, 34))
    vec_p = np.zeros((len(batch), max_num_seq, 34))
    mask_l = np.zeros((len(batch), max_num_atoms))
    mask_p = np.zeros((len(batch), max_num_seq))
    #adj_mat1 = np.zeros((len(batch), max_num_atoms, max_num_atoms))
    #adj_mat2 = np.zeros((len(batch), max_num_seq, max_num_seq))
    adj_inter = np.zeros((len(batch), max_num_atoms, max_num_seq))
    values = np.zeros((len(batch),))

    for i in range(len(batch)):
        n_latom = len(batch[i]['vec_latoms'])
        n_patom = len(batch[i]['vec_patoms'])

        vec_l[i, :n_latom] = batch[i]['vec_latoms']
        vec_p[i, :n_patom] = batch[i]['vec_patoms']
        #adj1 = batch[i]['adj1']
        adj_inter_i = batch[i]['adj_inter']
        """
        U, s, Vh = linalg.svd(adj_inter_i)#, full_matrices=False)
        adj_mat1[i, :n_latom, :n_latom] = U#*np.sqrt(s)# n_latom, n_latom
        adj_mat2[i, :n_patom, :n_patom] = Vh#*np.sqrt(s)# n_patom, n_latom
        """
        adj_inter[i, :n_latom, :n_patom] = adj_inter_i
        values[i] = batch[i]['values']
        mask_l[i, :n_latom ] = np.ones([n_latom])
        mask_p[i, :n_patom] = np.ones([n_patom])

    vec_l = torch.from_numpy(vec_l).float().to(device)#Variable(torch.LongTensor(vectors)).cuda(cudaDevid)#float does not fit for embedding
    vec_p = torch.from_numpy(vec_p).float().to(device)
    #adj_mat1 = torch.from_numpy(adj_mat1).float().to(device)
    #adj_mat2 = torch.from_numpy(adj_mat2).float().to(device)
    adj_inter = torch.from_numpy(adj_inter).float().to(device)
    mask_l = torch.from_numpy(mask_l).float().to(device)
    mask_p = torch.from_numpy(mask_p).float().to(device)
    values = torch.from_numpy(values).float().to(device)

    return vec_l, vec_p, adj_inter, mask_l, mask_p, values#adj_mat1, adj_mat2,


# class DTISampler(Sampler):
#     def __init__(self, weights, num_samples, replacement=True):
#         self.weights = np.array(weights) / np.sum(weights)
#         self.num_samples = num_samples
#         self.replacement = replacement
#
#     def __iter__(self):
#         retval = np.random.choice(len(self.weights), self.num_samples, replace=self.replacement, p=self.weights)
#         return iter(retval.tolist())
#
#     def __len__(self):
#         return self.num_samples


# # separate data into train and valid
# def div_tr_valid(data_pack):
#     allIdx = np.arange(len(data_pack[0]), dtype=np.int32)  # for index split process
#     train_Idx, valid_Idx, train_y, valid_y = train_test_split(allIdx, value, test_size=0.1)  # randomized
#     return train_Idx, valid_Idx


if __name__ == "__main__":
    # evaluate scheme
    n_epoch = 300
    usePickledData = True#False#
    batch_size = 64
    # print evaluation scheme
    print('Number of epochs:', n_epoch)
    # print('Hyper-parameters:', [para_names[i] + ':' + str(params[i]) for i in range(7)])

    # load data
    if usePickledData:
        datapack_kikd, datapack_ic50, datapack_test = pickle.load(open('prevData.pkl', 'rb'))
        dicAtom2I = pickle.load(open('prevData_dicts.pkl', 'rb'))
    else:
        dicAtom2I = defaultdict(lambda: len(dicAtom2I) + 1)
        datapack_kikd, datapack_ic50, datapack_test = load_data2(dicAtom2I)
        pickle.dump((datapack_kikd, datapack_ic50, datapack_test), open('prevData.pkl', 'wb'))  # , protocol=0)
        pickle.dump(dict(dicAtom2I), open('prevData_dicts.pkl', 'wb'))

    whole_len = len(datapack_kikd)
    valid_len = int(whole_len*0.03)
    train_dataset = MolDataset(datapack_kikd)
    train_kikd, valid_kikd = torch.utils.data.random_split(train_dataset, [whole_len - valid_len, valid_len])
    train_ic50 = MolDataset(datapack_ic50)
    test_data = MolDataset(datapack_test)

    tr_kikd_dl = DataLoader(train_kikd, batch_size, shuffle=True, collate_fn=collate_fn)#num_workers=7,
    tr_ic50_dl = DataLoader(train_ic50, batch_size, shuffle=True, collate_fn=collate_fn)
    valid_kikd_dl = DataLoader(valid_kikd, batch_size, shuffle=False, collate_fn=collate_fn)
    test_dl = DataLoader(test_data, batch_size, shuffle=False, collate_fn=collate_fn)

    print('train of kikd num:', whole_len - valid_len, 'valid num:', valid_len,
          'test num:', len(test_data))#+ len(train_ic50[0])
    train_and_eval(tr_kikd_dl, valid_kikd_dl, tr_ic50_dl, test_dl, len(dicAtom2I.keys()) + 1, batch_size, n_epoch,
                   loadFromSavedMoel=False, isTrainProcess=True)




