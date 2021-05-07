import math
import time
import pickle
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from pdbbind_utils_e2e import *

embedding_dim = 128
MAX_NUM_SEQ = 2000
MAX_NUM_ATOMS = 200
MAX_CONFS = 30


# simple self-attention module
def scaleDotProductAttention(q, k, v, mask=None):
    dk = k.shape[-1]
    sqrt_dk = 1 # dk ** 0.5
    attn = (torch.matmul(q, k.transpose(-2, -1))) / sqrt_dk
    if mask is not None:  # if x is (1, 2, 0, 0), mask is (1, 1, 0, 0)
        mask = mask > 0  # after torch 1.4, returns boolean type tensor
        attn = attn.masked_fill(~mask, -1e9)
        # softmax is normalized on the last axis (seq_len_k) so that the scores
    attn = F.softmax(attn, dim=-1)
    output = torch.matmul(attn, v)
    return output


class Net(nn.Module):
    def __init__(self, atom_vocab_size):
        super(Net, self).__init__()
        self.embed_atom1 = nn.Embedding(atom_vocab_size, embedding_dim)
        self.FC1 = nn.Linear(embedding_dim, embedding_dim)
        self.gat_lp = nn.ModuleList([GATgate_lp(embedding_dim) for _ in range(3)])
        self.gat2 = GATgate_lp2(embedding_dim)
        self.dropout1 = 0.3
        self.dropout2 = 0.3
        self.dropout3 = 0.3

        self.FC2 = nn.Linear(MAX_NUM_SEQ, 1)
        self.FC3 = nn.Linear(MAX_NUM_ATOMS, 1)

        self.FC4 = nn.Linear(embedding_dim, 1)
        self.FC5 = nn.Linear(MAX_NUM_ATOMS, 1)

        self.l_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.l_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.l_k = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.FC_final = nn.Linear(4, 1)
        self.ffLR = nn.LeakyReLU()

    def graphProcess_lp(self, vec_l, vec_p, adj_inter, isTrain=False):
        for k in range(len(self.gat_lp)):
            vec_l, vec_p = self.gat_lp[k](vec_l, vec_p, adj_inter)
            vec_l = F.dropout(vec_l, p=self.dropout2, training=isTrain)
            vec_p = F.dropout(vec_p, p=self.dropout2, training=isTrain)
        return vec_l, vec_p

    def SAProcess_l(self, vec_l1, isTrain=False):  # self-attention
        SA_l = vec_l1
        for k in range(1):
            SA_l = scaleDotProductAttention(self.l_q(SA_l), self.l_k(SA_l), self.l_v(SA_l), None)
        return SA_l

    def FC_lp_process(self, batch_size, vec_lp):
        vec_lp = self.FC2(vec_lp).view(batch_size, MAX_NUM_ATOMS)
        vec_lp = self.ffLR(vec_lp)
        vec_lp = F.dropout(vec_lp, p=self.dropout3, training=self.training)
        vec_lp = self.FC3(vec_lp)  # .view(batch_size, -1)
        return vec_lp  # batch, 1

    def FC_l_process(self, batch_size, vec_l):
        vec_l = self.FC4(vec_l).view(batch_size, MAX_NUM_ATOMS)
        vec_l = self.ffLR(vec_l)
        vec_l = F.dropout(vec_l, p=self.dropout3, training=self.training)
        vec_l = self.FC5(vec_l)  # .view(batch_size, -1)
        return vec_l  # batch, 1

    # vec_rdkit is features of 170 length for ligand
    def forward(self, vec_l1, vec_p1, adj_inter1, isTrain=False):
        batch_size = vec_l1.size(0)
        vec_l1 = self.embed_atom1(vec_l1)  # batch, max_num_atoms, embedding_dim
        vec_p1 = self.embed_atom1(vec_p1)  # batch, max_num_seq, embedding_dim

        # ------ BGNN ------
        vec_l = self.FC1(vec_l1)
        vec_l = F.dropout(vec_l, p=self.dropout1, training=isTrain)
        vec_p = self.FC1(vec_p1)
        #vec_p = F.dropout(vec_p, p=self.dropout1, training=isTrain)

        # BGNN
        vec_l, vec_p = self.graphProcess_lp(vec_l, vec_p, adj_inter1, isTrain)
        vec_lp = self.gat2(vec_l, vec_p, adj_inter1)  # batch, max_num_atoms, max_num_seq
        # ------ BGNN ------

        # SA for ligand
        lig_SA = self.SAProcess_l(vec_l1, isTrain)

        # first dr: sum
        pred_logit_sum = torch.sum(vec_lp, (-1, -2)).view(batch_size, 1)  # batch, 1
        lig_SA_sum = torch.sum(lig_SA, (-1, -2)).view(batch_size, 1)  # batch, 1
        # second dr: FC
        pred_logit_fc = self.FC_lp_process(batch_size, vec_lp)  # batch, 1, 1
        lig_SA_fc = self.FC_l_process(batch_size, lig_SA)

        # pred_logit_sum, lig_SA_sum
        _pred_logit = torch.cat((pred_logit_sum, pred_logit_fc, lig_SA_sum, lig_SA_fc), -1)  # batch, 4
        _pred_logit = self.FC_final(_pred_logit).view(batch_size)  # batch
        return _pred_logit


class GATgate_lp2(nn.Module):  # node merging process
    def __init__(self, n_dim):
        super(GATgate_lp2, self).__init__()
        self.w_l = nn.Linear(n_dim, n_dim)
        self.w_p = nn.Linear(n_dim, n_dim)
        self.LR = nn.LeakyReLU()

    def forward(self, vec_l, vec_p, adj_inter):  #
        h_l = self.w_l(vec_l)
        h_p = self.w_p(vec_p)

        # matmul
        intermat = torch.einsum('aij,ajk->aik',
                                (h_l, h_p.transpose(-1, -2)))  # batch, max_num_atoms, max_num_seq
        intermat = intermat * adj_inter  # batch, max_num_atoms, max_num_seq
        return intermat


class GATgate_lp(nn.Module):  # node update process
    def __init__(self, n_dim):
        super(GATgate_lp, self).__init__()
        self.w_l1 = nn.Linear(n_dim, n_dim)
        self.w_l2 = nn.Linear(n_dim, n_dim)
        self.w_p1 = nn.Linear(n_dim, n_dim)
        self.w_p2 = nn.Linear(n_dim, n_dim)
        self.LR = nn.LeakyReLU()

    def forward(self, vec_l, vec_p, adj_inter):
        h_l = self.w_l1(vec_l)
        h_p = self.w_p1(vec_p)

        # matmul
        h_l2 = torch.einsum('aij,ajk->aik', (adj_inter, h_p))  # batch, num_atoms, embedding
        h_l2 = self.LR(self.w_l2(h_l2 * h_l))  # batch, num_atoms, embedding

        # matmul
        h_p2 = torch.einsum('aij,ajk->aik',
                            (adj_inter.transpose(-1, -2), h_l))  # batch, num_seq, embedding
        h_p2 = self.LR(self.w_p2(h_p2 * h_p))  # batch, num_seq, embedding
        return h_l2, h_p2
