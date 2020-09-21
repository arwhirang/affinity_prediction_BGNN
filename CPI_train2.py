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
from rdkit import Chem

from pdbbind_utils2 import *
from CPI_model2 import *

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)  # <== not working on cuda 2... only cuda 0?

cudaDevid = 2
max_num_seq = 1000
max_num_atoms = 150
max_num_bonds = 100
embedsize = 200


# function for generating batch data
def batch_data_process(data, values):
    vecOfLatoms, fb, anb, bnb, nbs_mat, vecOfPatoms, tmpFRIMat, pairwise_mat, pairwise_exist = data

    L_mask = get_mask(vecOfLatoms, max_num_atoms)
    P_mask = get_mask(vecOfPatoms, max_num_seq)

    # embedding selection function requires 1-D indicies
    anb = add_index(anb, max_num_atoms)
    bnb = add_index(bnb, max_num_bonds)

    # convert to torch cuda data type
    L_mask = Variable(torch.FloatTensor(L_mask)).cuda(cudaDevid)
    vecOfLatoms = Variable(torch.LongTensor(vecOfLatoms)).cuda(cudaDevid)
    fb = Variable(torch.LongTensor(fb)).cuda(cudaDevid)
    anb = Variable(torch.LongTensor(anb)).cuda(cudaDevid)
    bnb = Variable(torch.LongTensor(bnb)).cuda(cudaDevid)
    nbs_mat = Variable(torch.LongTensor(nbs_mat)).cuda(cudaDevid)

    P_mask = Variable(torch.FloatTensor(P_mask)).cuda(cudaDevid)
    vecOfPatoms = Variable(torch.LongTensor(vecOfPatoms)).cuda(cudaDevid)
    tmpFRIMat = Variable(torch.FloatTensor(tmpFRIMat)).cuda(cudaDevid)
    values = torch.FloatTensor(values).cuda(cudaDevid)
    pairwise_label = torch.FloatTensor(pairwise_mat).cuda(cudaDevid)
    pairwise_exist = torch.FloatTensor(pairwise_exist).cuda(cudaDevid)

    return L_mask, vecOfLatoms, fb, anb, bnb, nbs_mat, vecOfPatoms, P_mask, tmpFRIMat, values, pairwise_label, pairwise_exist


# train and evaluate
def train_and_eval(train_data, train_y, valid_data, valid_y, test_data, test_y, vocab_atom_size, vocab_bond_size, batch_size=32, num_epoch=30, loadFromSavedMoel=False):
    net = Net(vocab_atom_size, vocab_bond_size)
    net.cuda(cudaDevid)

    if not loadFromSavedMoel:
        net.apply(weights_init)
    else:
        net.load_state_dict(torch.load("./current_model.pt"))

    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('total num params', pytorch_total_params)

    criterion1 = nn.MSELoss()
    criterion2 = nn.MSELoss()#Masked_BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0005, weight_decay=0,
                           amsgrad=False)  # True)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    min_rmse = 1000
    perf_name = ['RMSE', 'Pearson', 'Spearman', 'avg pairwise AUC']
    if not loadFromSavedMoel:
        for epoch in range(num_epoch):
            total_loss = 0
            affinity_loss = 0
            pairwise_loss = 0
            net.train()
            for i in range(int(len(train_data[0]) / batch_size)):
                if i % 100 == 0:
                    print('epoch', epoch, 'batch', i)

                currBatchTrain = [train_data[j][i * batch_size: (i + 1) * batch_size] for j in range(9)]
                currBatchTrain_y = train_y[i * batch_size: (i + 1) * batch_size]
                L_mask, vecOfLatoms, fb, anb, bnb, nbs_mat, vecOfPatoms, P_mask, tmpFRIMat, values, pairwise_label, pairwise_exist = batch_data_process(currBatchTrain, currBatchTrain_y)

                optimizer.zero_grad()
                affinity_pred, pairwise_pred = net(L_mask, vecOfLatoms, fb, anb, bnb, nbs_mat, vecOfPatoms, P_mask, tmpFRIMat)
                assert len(affinity_pred) == len(values)
                loss_aff = criterion1(affinity_pred, values)
                loss_pairwise = criterion2(pairwise_pred, pairwise_label)#, L_mask, P_mask, pairwise_exist)

                # rak ver
                loss = loss_aff + loss_pairwise

                total_loss += float(loss.data * batch_size)
                affinity_loss += float(loss_aff.data * batch_size)
                pairwise_loss += float(loss_pairwise.data * batch_size)

                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5)
                optimizer.step()
            # scheduler.step()

            loss_list = [total_loss, affinity_loss , pairwise_loss]
            loss_name = ['total loss', 'affinity loss', 'pairwise loss']
            print_loss = [loss_name[i] + ' ' + str(round(loss_list[i] / float(len(train_data)), 6)) for i in
                          range(len(loss_name))]
            print('epoch:', epoch, ' '.join(print_loss))

            net.eval()
            with torch.no_grad():
                valid_performance, valid_label, valid_output = test(net, valid_data, valid_y, batch_size)
                print_perf = [perf_name[i] + ' ' + str(round(valid_performance[i], 6)) for i in range(len(perf_name))]
                print('valid', len(valid_output), ' '.join(print_perf))

                if valid_performance[0] < min_rmse:
                    min_rmse = valid_performance[0]
                    torch.save(net.state_dict(), "./current_model.pt")
                    print('model saved <== current model is better than min_rmse')

    # test phase
    net.load_state_dict(torch.load("./current_model.pt"))
    net.eval()
    with torch.no_grad():
        test_performance, test_label, test_output = test(net, test_data, test_y, batch_size, isTest=True)
        print_perf = [perf_name[i] + ' ' + str(round(test_performance[i], 6)) for i in range(len(perf_name))]
        print('test ', len(test_output), ' '.join(print_perf))
    
    print('Finished.')
    return test_performance, test_label, test_output


def test(net, test_data, test_labels, batch_size, isTest=False):
    output_list = []
    label_list = []
    pairwise_auc_list = []
    for i in range(int(len(test_data[0]) / batch_size)):
        currBatchDat = [test_data[j][i * batch_size: (i + 1) * batch_size] for j in range(9)]
        currBatchVal = test_labels[i * batch_size: (i + 1) * batch_size]

        val_L_mask, val_vecOfLatoms, fb, anb, bnb, nbs_mat, val_vecOfPatoms, val_P_mask, val_tmpFRIMat, val_value, val_pairwise_label, val_pairwise_exist = \
            batch_data_process(currBatchDat, currBatchVal)
        affinity_pred, pairwise_pred = net(val_L_mask, val_vecOfLatoms, fb, anb, bnb, nbs_mat, val_vecOfPatoms, val_P_mask, val_tmpFRIMat)

        output_list += affinity_pred.cpu().detach().numpy().reshape(-1).tolist()
        label_list += val_value.reshape(-1).tolist()

        for j in range(len(val_pairwise_exist)):
            if val_pairwise_exist[j]:
                num_vertex = min(int(torch.sum(val_L_mask[j, :])), max_num_atoms)
                num_residue = min(int(torch.sum(val_P_mask[j, :])), max_num_seq)


                # print("1", num_vertex, num_residue)
                # isBreak = False
                # for ij in range(max_num_atoms):
                #     for jk in range(max_num_seq):
                #         if val_pairwise_label[j][ij][jk] == 1:
                #             print("2", ij, jk)
                #             isBreak = True
                #             break
                #     if isBreak:
                #         break
                # print(val_vecOfLatoms[j, :])
                # print(val_L_mask[j, :])

                pairwise_pred_j = pairwise_pred[j, :num_vertex, :num_residue].cpu().detach().numpy().reshape(-1)
                pairwise_label_j = val_pairwise_label[j, :num_vertex, :num_residue].cpu().reshape(-1)
                pairwise_auc_list.append(roc_auc_score(pairwise_label_j, pairwise_pred_j))

    if isTest:
        fw = open("fuer pruefen", "w")
        for ijk, ele in enumerate(pairwise_label_j):
            if num_vertex*num_residue == ijk:
                break

            fw.write("".join([str(ele), str(pairwise_pred_j[ijk])]))
            fw.write("\n")
        fw.close()

    output_list = np.array(output_list)
    label_list = np.array(label_list)
    rmse_value, pearson_value, spearman_value = reg_scores(label_list, output_list)
    average_pairwise_auc = np.mean(pairwise_auc_list)
    test_performance = [rmse_value, pearson_value, spearman_value, average_pairwise_auc]
    return test_performance, label_list, output_list


if __name__ == "__main__":
    # evaluate scheme
    measure = sys.argv[1]  # IC50 or KIKD
    n_epoch = 500
    n_rep = 1  # 10
    usePickledData = True#False
    batch_size = 64
    assert measure in ['IC50', 'KIKD']
    #GNN_depth, inner_CNN_depth, DMA_depth = 4, 2, 2
    #k_head, kernel_size, hidden_size1, hidden_size2 = 1, 5, 128, 128  # 2, 7, 128, 128
    #para_names = ['GNN_depth', 'inner_CNN_depth', 'DMA_depth', 'k_head', 'kernel_size', 'hidden_size1', 'hidden_size2']
    #params = [GNN_depth, inner_CNN_depth, DMA_depth, k_head, kernel_size, hidden_size1, hidden_size2]

    # print evaluation scheme
    print('Dataset: PDBbind v2019 with measurement', measure)
    print('Number of epochs:', n_epoch)
    print('Number of repeats:', n_rep)
    #print('Hyper-parameters:', [para_names[i] + ':' + str(params[i]) for i in range(7)])

    # load data
    if usePickledData:
        data_pack = pickle.load(open('prevData.pkl', 'rb'))
        dicAtom2I, dicBond2I = pickle.load(open('prevData_dicts.pkl', 'rb'))
    else:
        dicAtom2I = defaultdict(lambda: len(dicAtom2I) + 1)
        dicBond2I = defaultdict(lambda: len(dicBond2I) + 1)
        data_pack = load_data2(measure, dicAtom2I, dicBond2I)
        pickle.dump(data_pack, open('prevData.pkl', 'wb'))#, protocol=0)

    vecOfLatoms, fb, anb, bnb, nbs_mat, listOfPatypes, pdbid, values, tmpFRIMat, pairwise_mat, pairwise_exist = zip(*data_pack)
    # make vectors from atoms - nums of atoms are limited
    vecOfPatoms = np.zeros([len(listOfPatypes), max_num_seq])
    for i, Patypes in enumerate(listOfPatypes):
        for j, Ptype in enumerate(Patypes):
            #avec_Latoms = char2indices(Latypes, dicC2I, max_num_atoms)#dicC2I is the vocabulary
            vecOfPatoms[i][j] = dicAtom2I[atom_features(Ptype)]

    if not usePickledData:
        pickle.dump((dict(dicAtom2I), dict(dicBond2I)), open('prevData_dicts.pkl', 'wb'))

    data_pack2 = (np.array(vecOfLatoms, dtype=np.int32), np.array(fb, dtype=np.int32), np.array(anb, dtype=np.int32),
                  np.array(bnb, dtype=np.int32), np.array(nbs_mat, dtype=np.int32), np.array(vecOfPatoms, dtype=np.int32),
                  np.array(tmpFRIMat, dtype=np.float32), np.array(pairwise_mat, dtype=np.int32),
                  np.array(pairwise_exist, dtype=np.int32))

    allIdx = np.arange(len(values), dtype=np.int32)#for index split process
    for a_rep in range(n_rep):
        train_Idx, test_Idx, train_y, test_y  = train_test_split(allIdx, values, test_size=0.1)  # randomized
        train_Idx, valid_Idx, train_y, valid_y = train_test_split(train_Idx, train_y, test_size=0.1)  # randomized

        train_data = [data_pack2[i][train_Idx] for i in range(9)]
        valid_data = [data_pack2[i][valid_Idx] for i in range(9)]
        test_data  = [data_pack2[i][test_Idx] for i in range(9)]

        # currBatchTrain = train_data[0:batch_size]
        # L_mask, vecOfLatoms, vecOfPatoms, P_mask, tmpFRIMat, value = batch_data_process(currBatchTrain)
        # print(vecOfLatoms)
        # print(value)

        print('train num:', len(train_data[0]), 'valid num:', len(valid_data[0]), 'test num:', len(test_data[0]))
        train_and_eval(train_data, train_y, valid_data, valid_y, test_data, test_y, len(dicAtom2I.keys()) + 1, len(dicBond2I.keys()) + 1,
                                                                   batch_size, n_epoch, loadFromSavedMoel=False)
        #test_performance, test_label, test_output =
