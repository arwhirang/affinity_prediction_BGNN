import sys
import math
import pickle
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from rdkit import Chem
from pdbbind_utils_e2e import *
from model2 import *

cudaDevid = 2
device = torch.device("cuda:" + str(cudaDevid) if torch.cuda.is_available() else "cpu")
print(device)

MAX_CONFS = 30
MAX_NUM_SEQ = 2000
MAX_NUM_ATOMS = 200


# function for generating batch data
def runTheModel(net, sample, criterion1, isTraining=False):  # indexForModel 0:KIKD, 1:IC50
    vec_l, vec_p, adj_inter, values, pdbids, n_confs_perbatch, values_perbatch = sample  # vec_rdkit,
    aff_pred = net(vec_l, vec_p, adj_inter, isTrain=isTraining)  # vec_rdkit,
    assert len(aff_pred) == len(values)

    loss_aff = criterion1(aff_pred, values)
    loss = loss_aff

    # pairwise_loss += float(loss_pairwise.data * batch_size)
    if isTraining:
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5)

    aff_pred = aff_pred.cpu().detach().numpy().reshape(-1).tolist()
    return aff_pred, float(loss_aff.data), 0


# train and evaluate
def train_and_eval(train_dl, valid_dl, transtrain_dl, transtest_dl, vocab_atom_size,
                   num_epoch=30, loadFromSavedMoel=False, isTrainProcess=True):
    net = Net(vocab_atom_size)
    net.cuda(cudaDevid)

    if not loadFromSavedMoel:
        net.apply(weights_init)
    else:
        net.load_state_dict(torch.load("./current_model.pt"))

    criterion1 = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0005, weight_decay=0., amsgrad=True)

    best_cor_rmsd = 0
    best_epoch = 0
    perf_name = ['RMSE', 'Spearman', ]  # 'avg pairwise AUC']
    if isTrainProcess:
        for epoch in range(num_epoch):
            total_loss = 0
            affinity_loss = 0

            net.train()
            # training
            for i_batch, sample in enumerate(train_dl):
                optimizer.zero_grad()

                _, affloss, corloss = runTheModel(net, sample, criterion1, isTraining=True)
                total_loss += affloss + corloss
                affinity_loss += affloss
                optimizer.step()

            loss_list = [total_loss, affinity_loss]
            loss_name = ['total loss', 'affinity loss']
            print_loss = [loss_name[i] + ' ' + str(loss_list[i] / len(train_dl)) for i in range(len(loss_name))]  #
            print('epoch:', epoch, ' '.join(print_loss))

            net.eval()
            with torch.no_grad():
                # # for ic50
                valid_performance, valid_output = test(net, valid_dl)
                print_perf = [perf_name[i] + ' ' + str(round(valid_performance[i], 6)) for i in range(len(perf_name))]
                print('valid', len(valid_output), ' '.join(print_perf))

                if valid_performance[1]/valid_performance[0] > best_cor_rmsd:
                    best_cor_rmsd = valid_performance[1]/valid_performance[0]
                    torch.save(net.state_dict(), "./current_model.pt")
                    best_epoch = epoch
                    print('model saved <== current model is better')

    # test phase for PDBbind data
    net.load_state_dict(torch.load("./current_model.pt"))
    print("loaded from the best epoch:", best_epoch)

    #################################
    # for the trnasfer learning only
    for epoch in range(50):
        total_loss = 0
        affinity_loss = 0
        net.train()
        for i_batch, sample in enumerate(transtrain_dl):
            optimizer.zero_grad()

            # for transfer learning, freeze other layers and unfreeze the self-attention layer only
            for param in net.parameters():
                param.requires_grad = False
            net.FC4.weight.requires_grad = True
            net.FC4.bias.requires_grad = True
            net.FC5.weight.requires_grad = True
            net.FC5.bias.requires_grad = True
            net.l_q.weight.requires_grad = True
            net.l_k.weight.requires_grad = True
            net.l_v.weight.requires_grad = True
            net.FC_final.weight.requires_grad = True
            net.FC_final.bias.requires_grad = True

            _, affloss, corloss = runTheModel(net, sample, criterion1, isTraining=True)
            total_loss += affloss + corloss
            affinity_loss += affloss
            optimizer.step()

        loss_list = [total_loss, affinity_loss]
        loss_name = ['total loss', 'affinity loss']
        print_loss = [loss_name[i] + ' ' + str(loss_list[i] / len(train_dl)) for i in range(len(loss_name))]
        print('traintest epoch:', epoch, ' '.join(print_loss))

        net.eval()
        with torch.no_grad():
            test_performance, test_output = test(net, transtest_dl)
            print_perf = [perf_name[i] + ' ' + str(round(test_performance[i], 6)) for i in range(len(perf_name))]
            print('real test ', len(test_output), ' '.join(print_perf))

    #################################
    net.eval()
    with torch.no_grad():
        test_performance, test_output = test(net, transtest_dl)
        print_perf = [perf_name[i] + ' ' + str(round(test_performance[i], 6)) for i in range(len(perf_name))]
        print('real test ', len(test_output), ' '.join(print_perf))

        # save the thyroid prediction reult
        test_output, value_list, pdbid_list = pred_thy(net, transtest_dl)
        fw = open("test_output", "w")
        for idx, ele in enumerate(test_output):
            fw.write(str(ele) + "," + str(value_list[idx]))
            fw.write("\n")
        fw.close()
    print('train and evaluation Finished.')


def test(net, test_data):  # sub process for rough evaluatoin and test
    output_list = []
    label_list = []
    criterionTest = nn.MSELoss()
    loss_val = 0

    for i_batch, sample in enumerate(test_data):
        vec_l, vec_p, adj_inter, values, pdbids, n_confs_perbatch, values_perbatch = sample
        aff_pred, affloss, corloss = runTheModel(net, sample, criterionTest, isTraining=False)

        aff_pred_perbatch = np.zeros(len(n_confs_perbatch))  # lenngth of batch
        previdx = 0
        _values = values.cpu().detach().numpy().reshape(-1).tolist()
        for i, nconf in enumerate(n_confs_perbatch):
            nconf_int = int(nconf)
            aff_pred_perbatch[i] = sum(aff_pred[previdx: previdx + nconf_int + 1]) / nconf
            values_perbatch[i] = sum(_values[previdx: previdx + nconf_int + 1]) / nconf
            previdx = previdx + nconf_int

        output_list += list(aff_pred_perbatch)
        label_list += list(values_perbatch)
        loss_val += affloss + corloss

    output_list = np.array(output_list)
    label_list = np.array(label_list)

    print("validation loss:", str(loss_val / len(test_data)))
    rmse_value, spearman_value = reg_scores(label_list, output_list)
    test_performance = [rmse_value, spearman_value, ]
    return test_performance, output_list


def pred_thy(net, test_data):  # sub process for test (no evaluation)
    output_list = []
    pdbid_list = []
    value_list = []

    for i_batch, sample in enumerate(test_data):
        vec_l, vec_p, adj_inter, values, pdbids, n_confs_perbatch, values_perbatch = sample
        aff_pred = net(vec_l, vec_p, adj_inter)
        assert len(aff_pred) == len(values)
        aff_pred_perbatch = np.zeros(len(n_confs_perbatch))  # lenngth of batch
        previdx = 0
        aff_pred = aff_pred.cpu().detach().numpy().reshape(-1).tolist()
        _values = values.cpu().detach().numpy().reshape(-1).tolist()
        pdbid_list += pdbids

        for i, nconf in enumerate(n_confs_perbatch):
            nconf_int = int(nconf)
            aff_pred_perbatch[i] = sum(aff_pred[previdx: previdx + nconf_int + 1]) / nconf
            values_perbatch[i] = sum(_values[previdx: previdx + nconf_int + 1]) / nconf
            previdx = previdx + nconf_int
        output_list += list(aff_pred_perbatch)
        value_list += list(values_perbatch)

    return output_list, value_list, pdbid_list


# -----------codes for torch-specific dataloader start -----------

class MolDataset(Dataset):
    def __init__(self, datalist):
        self.datalist = datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        vec_latoms, vec_patoms, adj_inter, values, pdbid = self.datalist[idx]
        sample = {
            'vec_latoms': vec_latoms,
            'vec_patoms': vec_patoms,
            'adj_inter': adj_inter,
            'values': values,
            'pdbid': pdbid,
        }
        return sample


def collate_fn(batch):  # collate means collect and combine
    n_confs_whole = 0
    for i in range(len(batch)):
        n_confs_whole += len(batch[i]['vec_latoms'])

    vec_l = np.zeros((n_confs_whole, MAX_NUM_ATOMS))
    vec_p = np.zeros((n_confs_whole, MAX_NUM_SEQ))
    adj_inter = np.zeros((n_confs_whole, MAX_NUM_ATOMS, MAX_NUM_SEQ))
    values = np.zeros((n_confs_whole,))
    n_confs_perbatch = np.zeros((len(batch),))
    values_perbatch = np.zeros((len(batch),))
    pdbids = ["0"]*len(batch)  # n_confs_whole

    idx = 0
    for i in range(len(batch)):
        n_conf = len(batch[i]['vec_latoms'])  # conf num is same for vec_latoms, vec_patoms, vec_rdkit, adj_inter
        n_confs_perbatch[i] = min(MAX_CONFS, n_conf)
        values_perbatch[i] = batch[i]['values']
        pdbids[i] = batch[i]['pdbid']
        for j in range(n_conf):
            if j == MAX_CONFS:  # no more than 30
                break
            n_latom = len(batch[i]['vec_latoms'][j])
            n_patom = len(batch[i]['vec_patoms'][j])
            vec_l[idx, :n_latom] = batch[i]['vec_latoms'][j]
            vec_p[idx, :n_patom] = batch[i]['vec_patoms'][j]
            adj_inter[idx, :n_latom, :n_patom] = batch[i]['adj_inter'][j]
            values[idx] = batch[i]['values']

            idx += 1

        if n_conf == 0:
            print("critical error found in data")
            print("require new preprocessing!", batch[i]['pdbid'])

    vec_l = torch.from_numpy(vec_l).long().to(device)
    vec_p = torch.from_numpy(vec_p).long().to(device)
    adj_inter = torch.from_numpy(adj_inter).float().to(device)
    values = torch.from_numpy(values).float().to(device)
    return vec_l, vec_p, adj_inter, values, pdbids, n_confs_perbatch, values_perbatch

# -----------codes for torch-specific dataloader end -----------


if __name__ == "__main__":
    n_epoch = 100
    usePickledData = True  # for fast loading of PDBbind data
    batch_size = 20
    print('Number of epochs:', n_epoch)

    # load data
    if usePickledData:
        datapack_kikd, datapack_ic50, datapack_test = pickle.load(open('prevData_IC50.pkl', 'rb'))
        dic_atom2i = pickle.load(open('prevData_IC50_dicts.pkl', 'rb'))
    else:
        dic_atom2i = {}
        datapack_kikd, datapack_ic50, datapack_test = load_data2(dic_atom2i)
        pickle.dump((datapack_kikd, datapack_ic50, datapack_test), open('prevData_IC50.pkl', 'wb'))  # , protocol=0)
        pickle.dump(dict(dic_atom2i), open('prevData_IC50_dicts.pkl', 'wb'))

    # for ic50_wholebace
    BACEset = loadgc4BACEset(dic_atom2i)
    test_BACEdata = MolDataset(BACEset)
    transtest_dl = DataLoader(test_BACEdata, batch_size, shuffle=False, collate_fn=collate_fn)
    chembl_bace = loadchemblBACEset(dic_atom2i)
    chembl_bace_trMD = MolDataset(chembl_bace)
    transtrain_dl = DataLoader(chembl_bace_trMD, batch_size, shuffle=True, collate_fn=collate_fn)
    # # for ic50_CatS
    # print("started reading CATS")
    # CATSset = loadwholeCATSset(dic_atom2i)
    # gc3CATSset = loadgc3_CATSset(dic_atom2i)  # additional data for transfer learning
    # train_CATSdata = MolDataset(gc3CATSset)
    # test_CATSdata = MolDataset(CATSset)
    # transtrain_dl = DataLoader(train_CATSdata, batch_size, shuffle=True, collate_fn=collate_fn)
    # transtest_dl = DataLoader(test_CATSdata, batch_size, shuffle=False, collate_fn=collate_fn)

    # ramdomly select valid data among train data (PDBbind training)
    whole_len = len(datapack_ic50)
    valid_len = int(whole_len * 0.07)
    train_dataset = MolDataset(datapack_ic50)
    train_MD, valid_MD = torch.utils.data.random_split(train_dataset, [whole_len - valid_len, valid_len])
    train_dl = DataLoader(train_MD, batch_size, shuffle=True, collate_fn=collate_fn)
    valid_dl = DataLoader(valid_MD, batch_size, shuffle=True, collate_fn=collate_fn)

    print('train of ic50 num:', whole_len - valid_len, 'valid num:', valid_len)
    print('num of atom vocabs:', len(dic_atom2i.keys()) + 1)
    train_and_eval(train_dl, valid_dl, transtrain_dl, transtest_dl, len(dic_atom2i.keys()) + 1, n_epoch,
                   loadFromSavedMoel=False, isTrainProcess=True)



