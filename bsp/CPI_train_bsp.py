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
from rdkit import Chem

from pdbbind_utils_bsp import *
from CPI_model_bsp import *

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)  # <== not working on cuda 2... only cuda 0?

cudaDevid = 2
max_num_seq = 1000
max_num_atoms = 150
max_num_bonds = 100
embedsize = 200


# function for generating batch data
def batch_data_process(data):
    vecOfLatoms, vecOfPatoms, pairwise_mat, pairwise_exist, value = data

    L_mask = get_mask(vecOfLatoms, max_num_atoms)
    P_mask = get_mask(vecOfPatoms, max_num_seq)

    # embedding selection function requires 1-D indicies
    # anb = adjMat2D(anb)
    # convert to torch cuda data type
    L_mask = Variable(torch.FloatTensor(L_mask)).cuda(cudaDevid)
    vecOfLatoms = Variable(torch.LongTensor(vecOfLatoms)).cuda(cudaDevid)
    P_mask = Variable(torch.FloatTensor(P_mask)).cuda(cudaDevid)
    vecOfPatoms = Variable(torch.LongTensor(vecOfPatoms)).cuda(cudaDevid)
    value = torch.FloatTensor(value).cuda(cudaDevid)
    pairwise_label = torch.FloatTensor(pairwise_mat).cuda(cudaDevid)
    pairwise_exist = torch.FloatTensor(pairwise_exist).cuda(cudaDevid)
    return L_mask, vecOfLatoms, P_mask, vecOfPatoms, value, pairwise_label, pairwise_exist


# function for generating batch data
def runTheModel(net, data, i, indexForModel, criterion1, criterion2, isTraining=False):  # indexForModel 0:KIKD, 1:IC50
    currBatchTrain = [data[j][i * batch_size: (i + 1) * batch_size] for j in range(5)]
    L_mask, vecOfLatoms, P_mask, vecOfPatoms, values, pairwise_label, pairwise_exist = \
        batch_data_process(currBatchTrain)

    affinity_pred, pairwise_pred = net(L_mask, vecOfLatoms, P_mask, vecOfPatoms, indexForModel, isTrain=isTraining)
    assert len(affinity_pred) == len(values)
    loss_aff = criterion1(affinity_pred, values)
    loss_pairwise = criterion2(pairwise_pred, pairwise_label)#, L_mask, P_mask, pairwise_exist)

    # rak ver
    loss = loss_pairwise# +loss_aff

    # pairwise_loss += float(loss_pairwise.data * batch_size)
    if isTraining:
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5)
    return affinity_pred, pairwise_pred, float(loss.data * batch_size), values


# train and evaluate
def train_and_eval(train_kikd, valid_kikd, train_ic50, test_data,
                   vocab_atom_size, batch_size=32, num_epoch=30, loadFromSavedMoel=False):
    net = Net(vocab_atom_size)
    net.cuda(cudaDevid)

    if not loadFromSavedMoel:
        net.apply(weights_init)
    else:
        net.load_state_dict(torch.load("./current_model.pt"))

    kikdIndex = torch.Tensor([1, 0]).cuda(cudaDevid)
    ic50Index = torch.Tensor([0, 1]).cuda(cudaDevid)

    # pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print('total num params', pytorch_total_params)
    criterion1 = nn.MSELoss()
    criterion2 = nn.MSELoss()#Masked_BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001, weight_decay=0,
                           amsgrad=False)  # True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    # min_rmse = 1000
    min_auc = 0
    perf_name = ['RMSE', 'Pearson', 'Spearman', 'avg pairwise AUC']
    if not loadFromSavedMoel:
        for epoch in range(num_epoch):
            total_loss = 0
            affinity_loss = 0
            pairwise_loss = 0
            net.train()

            kikdLen = int(len(train_kikd[0]) / batch_size)
            ic50Len = int(len(train_ic50[0]) / batch_size)
            lenTrain = max(kikdLen, ic50Len)
            for i in range(lenTrain):
                if i < kikdLen:
                    optimizer.zero_grad()
                    _, _, theLoss, _ = runTheModel(net, train_kikd, i, kikdIndex, criterion1, criterion2,
                                                            isTraining=True)
                    total_loss += theLoss
                    affinity_loss += theLoss
                    optimizer.step()

                if i < ic50Len:
                    optimizer.zero_grad()
                    _, _, theLoss, _ = runTheModel(net, train_ic50, i, ic50Index, criterion1, criterion2,
                                                            isTraining=True)
                    total_loss += theLoss
                    affinity_loss += theLoss
                    optimizer.step()

            scheduler.step()

            loss_list = [total_loss, affinity_loss, pairwise_loss]
            loss_name = ['total loss', 'affinity loss' , 'pairwise loss']
            print_loss = [loss_name[i] + ' ' + str(loss_list[i] / (len(train_kikd[0])+ len(train_ic50[0]))) for i in
                          range(len(loss_name))]  #
            print('epoch:', epoch, ' '.join(print_loss))

            net.eval()
            with torch.no_grad():
                valid_performance, valid_output = test(net, valid_kikd, kikdIndex, batch_size)
                print_perf = [perf_name[i] + ' ' + str(round(valid_performance[i], 6)) for i in range(len(perf_name))]
                print('valid', len(valid_output), ' '.join(print_perf))

                if valid_performance[3] > min_auc:#< min_rmse:
                    #min_rmse = valid_performance[0]
                    min_auc = valid_performance[3]
                    torch.save(net.state_dict(), "./current_model.pt")
                    print('model saved <== current model is better than min_rmse')

    # test phase
    net.load_state_dict(torch.load("./current_model.pt"))
    net.eval()
    with torch.no_grad():
        test_performance, test_output = test(net, test_data, kikdIndex, batch_size)
        print_perf = [perf_name[i] + ' ' + str(round(test_performance[i], 6)) for i in range(len(perf_name))]
        print('test ', len(test_output), ' '.join(print_perf))

    print('Finished.')


def test(net, test_data, kikdIndex, batch_size, isTest=False):  # only KIKD
    output_list = []
    label_list = []
    pairwise_auc_list = []
    criterionTest1 = nn.MSELoss()
    criterionTest2 = nn.MSELoss()
    loss_val = 0
    for i in range(int(len(test_data[0]) / batch_size)):
        currBatchTest = [test_data[j][i * batch_size: (i + 1) * batch_size] for j in range(5)]
        val_L_mask, _, val_P_mask, _, _, val_pairwise_label, val_pairwise_exist = \
            batch_data_process(currBatchTest)

        affinity_pred, pairwise_pred, theLoss, val_value = runTheModel(net, test_data, i, kikdIndex, criterionTest1,
                                                                       criterionTest2)

        output_list += affinity_pred.cpu().detach().numpy().reshape(-1).tolist()
        label_list += val_value.reshape(-1).tolist()
        loss_val += theLoss
        for j in range(len(val_pairwise_exist)):
            if val_pairwise_exist[j]:
                num_vertex = min(int(torch.sum(val_L_mask[j, :])), max_num_atoms)
                num_residue = min(int(torch.sum(val_P_mask[j, :])), max_num_seq)

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

    print("validation loss:", str(loss_val / len(test_data[0])))
    rmse_value, pearson_value, spearman_value = reg_scores(label_list, output_list)
    average_pairwise_auc = np.mean(pairwise_auc_list)
    test_performance = [rmse_value, pearson_value, spearman_value, average_pairwise_auc]
    return test_performance, output_list


# separate data into train and valid
def postproc_datapack(data_pack, isKIKD=False):
    # normalize values into 0 to 1 scale
    # values_norm = (values - np.min(values))/np.ptp(values)
    vecOfLatoms, vecOfPatoms, value, pairwise_mat, pairwise_exist = zip(*data_pack)

    # vec_all_atoms = np.array(vec_all_atoms, dtype=np.int32)
    data_pack2 = (np.array(vecOfLatoms, dtype=np.int32), np.array(vecOfPatoms, dtype=np.int32),
                  np.array(pairwise_mat, dtype=np.int32), np.array(pairwise_exist, dtype=np.int32),
                  np.array(value, dtype=np.float32))

    if isKIKD:
        allIdx = np.arange(len(data_pack2[0]), dtype=np.int32)  # for index split process
        train_Idx, valid_Idx, train_y, valid_y = train_test_split(allIdx, value, test_size=0.1)  # randomized
        train_data = [data_pack2[i][train_Idx] for i in range(5)]
        valid_data = [data_pack2[i][valid_Idx] for i in range(5)]
        return train_data, valid_data
    else:
        return data_pack2, None


if __name__ == "__main__":
    # evaluate scheme
    measure = sys.argv[1]  # IC50 or KIKD
    n_epoch = 100
    usePickledData = True# False
    batch_size = 64
    assert measure in ['IC50', 'KIKD']
    # GNN_depth, inner_CNN_depth, DMA_depth = 4, 2, 2
    # k_head, kernel_size, hidden_size1, hidden_size2 = 1, 5, 128, 128  # 2, 7, 128, 128
    # para_names = ['GNN_depth', 'inner_CNN_depth', 'DMA_depth', 'k_head', 'kernel_size', 'hidden_size1', 'hidden_size2']
    # params = [GNN_depth, inner_CNN_depth, DMA_depth, k_head, kernel_size, hidden_size1, hidden_size2]

    # print evaluation scheme
    print('Dataset: PDBbind v2019 with measurement', measure)
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

    train_kikd, valid_kikd = postproc_datapack(datapack_kikd, isKIKD=True)
    train_ic50, _ = postproc_datapack(datapack_ic50)
    test_data, _ = postproc_datapack(datapack_test)

    print('train num:', len(train_kikd[0]+ len(train_ic50[0])) , 'valid num:', len(valid_kikd[0]),
          'test num:', len(test_data[0]))#
    train_and_eval(train_kikd, valid_kikd, train_ic50, test_data, len(dicAtom2I.keys()) + 1, batch_size, n_epoch,
                   loadFromSavedMoel=False)
