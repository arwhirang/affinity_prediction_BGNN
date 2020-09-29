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
def batch_data_process(data):
    vecOfLatoms, anb, vecOfPatoms, pdbid, values, tmpFRIMat, pairwise_mat, pairwise_exist = data

    L_mask = get_mask(vecOfLatoms, max_num_atoms)
    P_mask = get_mask(vecOfPatoms, max_num_seq)
    # embedding selection function requires 1-D indicies
    anb = adjMat2D(anb)
    # convert to torch cuda data type
    L_mask = Variable(torch.FloatTensor(L_mask)).cuda(cudaDevid)
    vecOfLatoms = Variable(torch.LongTensor(vecOfLatoms)).cuda(cudaDevid)
    anb = Variable(torch.LongTensor(anb)).cuda(cudaDevid)
    P_mask = Variable(torch.FloatTensor(P_mask)).cuda(cudaDevid)
    vecOfPatoms = Variable(torch.LongTensor(vecOfPatoms)).cuda(cudaDevid)
    tmpFRIMat = Variable(torch.FloatTensor(tmpFRIMat)).cuda(cudaDevid)
    values = torch.FloatTensor(values).cuda(cudaDevid)
    pairwise_label = torch.FloatTensor(pairwise_mat).cuda(cudaDevid)
    pairwise_exist = torch.FloatTensor(pairwise_exist).cuda(cudaDevid)
    return L_mask, vecOfLatoms, anb, vecOfPatoms, P_mask, tmpFRIMat, values, pairwise_label, pairwise_exist


# function for generating batch data
def runTheModel(net, data, j, indexForModel):#indexForModel 0:KIKD, 1:IC50
    currBatchTrain = [data[j][i * batch_size: (i + 1) * batch_size] for j in range(8)]
    L_mask, vecOfLatoms, anb, vecOfPatoms, P_mask, tmpFRIMat, values, pairwise_label, pairwise_exist = batch_data_process(
        currBatchTrain)

    optimizer.zero_grad()
    affinity_pred = net(L_mask, vecOfLatoms, anb, vecOfPatoms, P_mask, tmpFRIMat, indexForModel)
    return affinity_pred

# train and evaluate
def train_and_eval(train_kikd, valid_kikd, train_ic50, valid_ic50, test_data, vocab_atom_size, vocab_bond_size, batch_size=32, num_epoch=30, loadFromSavedMoel=False):
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    min_rmse = 1000
    perf_name = ['RMSE', 'Pearson', 'Spearman', 'avg pairwise AUC']
    if not loadFromSavedMoel:
        for epoch in range(num_epoch):
            total_loss = 0
            affinity_loss = 0
            pairwise_loss = 0
            net.train()
            for i in range(int(len(train_data[0]) / batch_size)):

                assert len(affinity_pred) == len(values)
                loss_aff = criterion1(affinity_pred, values)
                # loss_pairwise = criterion2(pairwise_pred, pairwise_label)#, L_mask, P_mask, pairwise_exist)

                # rak ver
                loss = loss_aff# + loss_pairwise

                total_loss += float(loss.data * batch_size)
                affinity_loss += float(loss_aff.data * batch_size)
                # pairwise_loss += float(loss_pairwise.data * batch_size)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5)
                optimizer.step()
            scheduler.step()

            loss_list = [total_loss, affinity_loss ]#, pairwise_loss]
            loss_name = ['total loss', 'affinity loss']#, 'pairwise loss']
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
    criterionTest = nn.MSELoss()
    loss_val = 0
    for i in range(int(len(test_data[0]) / batch_size)):
        currBatchDat = [test_data[j][i * batch_size: (i + 1) * batch_size] for j in range(9)]
        currBatchVal = test_labels[i * batch_size: (i + 1) * batch_size]

        val_L_mask, val_vecOfLatoms, fb, anb, bnb, nbs_mat, val_vecOfPatoms, val_P_mask, val_tmpFRIMat, val_value, val_pairwise_label, val_pairwise_exist = \
            batch_data_process(currBatchDat, currBatchVal)
        affinity_pred = net(val_L_mask, val_vecOfLatoms, fb, anb, bnb, nbs_mat, val_vecOfPatoms, val_P_mask, val_tmpFRIMat)#, pairwise_pred

        output_list += affinity_pred.cpu().detach().numpy().reshape(-1).tolist()
        label_list += val_value.reshape(-1).tolist()
        loss_val += float(criterionTest(affinity_pred, val_value)* batch_size)
    #     for j in range(len(val_pairwise_exist)):
    #         if val_pairwise_exist[j]:
    #             num_vertex = min(int(torch.sum(val_L_mask[j, :])), max_num_atoms)
    #             num_residue = min(int(torch.sum(val_P_mask[j, :])), max_num_seq)
    #
    #             pairwise_pred_j = pairwise_pred[j, :num_vertex, :num_residue].cpu().detach().numpy().reshape(-1)
    #             pairwise_label_j = val_pairwise_label[j, :num_vertex, :num_residue].cpu().reshape(-1)
    #             pairwise_auc_list.append(roc_auc_score(pairwise_label_j, pairwise_pred_j))
    #
    # if isTest:
    #     fw = open("fuer pruefen", "w")
    #     for ijk, ele in enumerate(pairwise_label_j):
    #         if num_vertex*num_residue == ijk:
    #             break
    #
    #         fw.write("".join([str(ele), str(pairwise_pred_j[ijk])]))
    #         fw.write("\n")
    #     fw.close()

    output_list = np.array(output_list)
    label_list = np.array(label_list)

    print("validation loss:", loss_val)
    rmse_value, pearson_value, spearman_value = reg_scores(label_list, output_list)
    #average_pairwise_auc = np.mean(pairwise_auc_list)
    average_pairwise_auc = 0
    test_performance = [rmse_value, pearson_value, spearman_value, average_pairwise_auc]
    return test_performance, label_list, output_list


# separate data into train and valid
def postproc_datapack(data_pack):

    # normalize values into 0 to 1 scale
    # values_norm = (values - np.min(values))/np.ptp(values)

    vecOfLatoms, anb, vecOfPatoms, pdbid, values, tmpFRIMat, pairwise_mat, pairwise_exist = zip(*data_pack)
    allIdx = np.arange(len(values), dtype=np.int32)  # for index split process
    train_Idx, valid_Idx, train_y, valid_y = train_test_split(allIdx, values, test_size=0.1)  # randomized

    train_data = [data_pack[i][train_Idx] for i in range(8)]
    valid_data = [data_pack[i][valid_Idx] for i in range(8)]
    return train_data, valid_data


if __name__ == "__main__":
    # evaluate scheme
    measure = sys.argv[1]  # IC50 or KIKD
    n_epoch = 300
    usePickledData = False#True
    batch_size = 128
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
        datapack_kikd, datapack_ic50, datapack_test = pickle.load(open('prevData.pkl', 'rb'))
        dicAtom2I, dicBond2I = pickle.load(open('prevData_dicts.pkl', 'rb'))
    else:
        dicAtom2I = defaultdict(lambda: len(dicAtom2I) + 1)
        dicBond2I = defaultdict(lambda: len(dicBond2I) + 1)
        datapack_kikd, datapack_ic50, datapack_test = load_data2(measure, dicAtom2I, dicBond2I)
        pickle.dump((datapack_kikd, datapack_ic50, datapack_test), open('prevData.pkl', 'wb'))#, protocol=0)
        pickle.dump((dict(dicAtom2I), dict(dicBond2I)), open('prevData_dicts.pkl', 'wb'))

    train_kikd, valid_kikd = postproc_datapack(datapack_kikd)
    train_ic50, valid_ic50 = postproc_datapack(datapack_ic50)




    # currBatchTrain = train_data[0:batch_size]
    # L_mask, vecOfLatoms, vecOfPatoms, P_mask, tmpFRIMat, value = batch_data_process(currBatchTrain)
    # print(vecOfLatoms)
    # print(value)

    print('train num:', len(train_kikd[0]) + len(train_ic50[0]), 'valid num:', len(valid_kikd[0]) + len(valid_ic50[0]), 'test num:', len(datapack_test[0]))
    train_and_eval(train_kikd, valid_kikd, train_ic50, valid_ic50, test_data, len(dicAtom2I.keys()) + 1, len(dicBond2I.keys()) + 1,
                                                               batch_size, n_epoch, loadFromSavedMoel=False)
    #test_performance, test_label, test_output =
