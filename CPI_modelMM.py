import math
import time
import pickle
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.nn.utils import weight_norm
# from sru import SRU, SRUCell
from pdbbind_utilsMM import *
from MultiHeadAttention import *

# some predefined parameters
# elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K',
#              'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In',
#              'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U',
#              'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs', 'unknown']
# atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
# bond_fdim = 6
embedding_dim = 128
dff_ = 1024
max_num_seq = 2000
max_num_atoms = 200


# define the model
def scaleDotProductAttention(q, k, v, mask=None):
    dk = k.shape[-1]
    sqrt_dk = 1#dk ** 0.5
    attn = (torch.matmul(q, k.transpose(-2, -1))) / sqrt_dk
    if mask is not None:  # if x is (1, 2, 0, 0), mask is (1, 1, 0, 0)
        mask = mask > 0  # after torch 1.4, returns boolean type tensor
        attn = attn.masked_fill(~mask, -1e9)
        # softmax is normalized on the last axis (seq_len_k) so that the scores
    attn = F.softmax(attn, dim=-1)

    output = torch.matmul(attn, v)
    return output


class CustomRSum(nn.Module):
    def __init__(self):
        super(CustomRSum, self).__init__()

    def forward(self, inputs, dWhich):
        return torch.sum(inputs * dWhich, dim=1)  # only the 1 instnce survives


class Net(nn.Module):
    def __init__(self, atom_vocab_size):
        super(Net, self).__init__()
        self.embed_atom = nn.Embedding(atom_vocab_size, embedding_dim)
        self.FC1 = nn.Linear(34, embedding_dim)
        # self.avp1d = nn.AvgPool1d(embedding_dim)
        self.FC3_1 = nn.Linear(embedding_dim, 1)#max_num_atoms
        self.FC3_2 = nn.Linear(embedding_dim, 1)
        # self.mu = nn.Parameter(torch.Tensor([4]).float())
        # self.dev = nn.Parameter(torch.Tensor([1]).float())
        self.FC_kikd = nn.ModuleList([
            nn.Linear(max_num_atoms+max_num_seq, embedding_dim) if i == 0 else
            nn.Linear(embedding_dim, 1) if i == 2 else
            nn.Linear(embedding_dim, embedding_dim) for i in range(3)])
        self.FC_ic50 = nn.ModuleList(
            [nn.Linear(max_num_atoms+max_num_seq, embedding_dim) if i == 0 else
             nn.Linear(embedding_dim, 1) if i == 2 else
             nn.Linear(embedding_dim, embedding_dim) for i in range(3)])
        """Graph whatever"""
        # self.gats = nn.ModuleList([GATgate(embedding_dim) for _ in range(3)])
        self.gat_lp = nn.ModuleList([GATgate_lp(embedding_dim) for _ in range(3)])
        self.gat2 = GATgate_lp2(embedding_dim)
        self.ffn = nn.Sequential(nn.Linear(embedding_dim, dff_),
                                 nn.ReLU(),  # (batch_size, seq_len, dff)
                                 nn.Linear(dff_, embedding_dim)  # (batch_size, seq_len, d_model)
                                 )
        self.ffn2 = nn.Sequential(nn.Linear(embedding_dim, dff_),
                                  nn.ReLU(),  # (batch_size, seq_len, dff)
                                  nn.Linear(dff_, embedding_dim)  # (batch_size, seq_len, d_model)
                                  )
        self.dropout1 = 0.0
        self.dropout2 = 0.1
        """selfatt_l"""
        self.l_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.l_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.l_k = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # self.layernorm1 = nn.LayerNorm(embedding_dim)
        # self.layernorm2 = nn.LayerNorm(embedding_dim)
        """selfatt_p"""
        self.p_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.p_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.p_k = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.ffLR = nn.LeakyReLU()

    # def pairwise_pred_module(self, batch_size, pairwise_pred, lig_mask, poc_mask):
    #     pairwise_pred = pairwise_pred*self.outFRI
    #     pairwise_conv_res = self.pairwise_conv(pairwise_pred.view(-1, 1, max_num_atoms, max_num_seq))# batch, 16, max_num_atoms, max_num_seq
    #     pairwise_conv_res = torch.transpose(pairwise_conv_res, 1, 2)
    #     pairwise_conv_res = torch.transpose(pairwise_conv_res, 2, 3)# batch, max_num_atoms, max_num_seq, 16
    #     pairwise_pred_fin = self.FCpip_pairwise(pairwise_conv_res)# batch, max_num_atoms, max_num_seq, 1
    #
    #     pairwise_mask = torch.matmul(lig_mask.view(batch_size, -1, 1),
    #                                  poc_mask.view(batch_size, 1, -1))  # batch, max_num_atoms, max_num_seq
    #     pairwise_pred_fin = pairwise_pred_fin.view(-1, max_num_atoms, max_num_seq).contiguous() * pairwise_mask# batch, max_num_atoms, max_num_seq
    #     return pairwise_pred_fin


    def graphProcess_lp(self, vec_l, vec_p, adj_inter, isTrain=False):
        for k in range(len(self.gat_lp)):
            vec_l, vec_p = self.gat_lp[k](vec_l, vec_p, adj_inter)
            vec_l = F.dropout(vec_l, p=self.dropout2, training=isTrain)
            vec_p = F.dropout(vec_p, p=self.dropout2, training=isTrain)
        return vec_l, vec_p

    def fully_connected_kikd(self, hidden):
        for k in range(len(self.FC_kikd)):
            if k < len(self.FC_kikd) - 1:
                hidden = self.FC_kikd[k](hidden)
                hidden = F.dropout(hidden, p=self.dropout2, training=self.training)
            else:
                hidden = self.FC_kikd[k](hidden)
        hidden = self.ffLR(hidden)
        return hidden

    def fully_connected_ic50(self, hidden):
        for k in range(len(self.FC_ic50)):
            if k < len(self.FC_ic50) - 1:
                hidden = self.FC_ic50[k](hidden)
                hidden = F.dropout(hidden, p=self.dropout2, training=self.training)
            else:
                hidden = self.FC_ic50[k](hidden)
        hidden = self.ffLR(hidden)
        return hidden


    def forward(self, vec_l, vec_p, adj_inter, mask_l, mask_p, indexForModel, isTrain=False):
        batch_size = vec_l.size(0)
        # vectors = self.embed_atom(vectors)  # batch, max_num_atoms+max_num_seq, embedding_dim
        vec_l = self.FC1(vec_l)
        vec_l = F.dropout(vec_l, p=self.dropout1, training=isTrain)
        vec_p = self.FC1(vec_p)
        vec_p = F.dropout(vec_p, p=self.dropout1, training=isTrain)
        vec_l, vec_p = self.graphProcess_lp(vec_l, vec_p, adj_inter, isTrain)

        #vec_l = torch.matmul(adj_l, vec_l)# batch, max_num_atoms, embedding_dim
        #vec_l = scaleDotProductAttention(self.l_q(vec_l), self.l_k(vec_l), self.l_v(vec_l), mask_l.view(batch_size, -1, 1))
        # vec_l = self.ffn(vec_l)
        # vec_l = F.dropout(vec_l, p=self.dropout2, training=isTrain)

        #vec_p = torch.matmul(adj_p, vec_p)  # batch, max_num_atoms, embedding_dim
        # vec_p = scaleDotProductAttention(self.p_q(vec_p), self.p_k(vec_p), self.p_v(vec_p), mask_p.view(batch_size, -1, 1))
        # vec_p = self.ffn2(vec_p)
        # vec_p = F.dropout(vec_p, p=self.dropout2, training=isTrain)
        vec_lp = self.gat2(vec_l, vec_p, adj_inter)
        #vec_l = self.FC3_1(vec_l).view(batch_size, -1)  # batch, embedding_dim
        #vec_p = self.FC3_2(vec_p).view(batch_size, -1)  # batch, embedding_dim

        #finStage_kikd = torch.cat((vec_l, vec_p), -1).view(batch_size, -1)
        finStage_kikd = torch.sum(vec_lp, (1, 2)).view(batch_size, -1)#self.fully_connected_kikd(finStage_kikd)  # batch, 1

        #finStage_ic50 = torch.cat((vec_l, vec_p), -1).view(batch_size, -1)  # batch,(embedding_dim+ max_num_atoms+max_num_seq)
        finStage_ic50 = torch.sum(vec_lp, (1, 2)).view(batch_size, -1)#self.fully_connected_ic50(finStage_ic50)  # batch, 1

        # distinguish ic50 and the kikd
        x_out = torch.cat((finStage_kikd, finStage_ic50), -1)  # batch, 2
        pred_logit = CustomRSum()(x_out, indexForModel)
        return pred_logit.view(-1), pred_logit


# class GATgate(torch.nn.Module):
#     def __init__(self, n_out_feature):
#         super(GATgate, self).__init__()
#         self.n_out = n_out_feature
#         # self.make_dim_same = nn.Linear(n_in_feature, n_out_feature)
#         self.W = nn.Linear(n_out_feature, n_out_feature)
#         self.A = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))
#         self.gate = nn.Linear(n_out_feature * 2, 1)
#
#     def forward(self, x, adj):
#         # if self.n_in != self.n_out:
#         #    x = self.make_dim_same(x)
#         h = self.W(x)
#         e = torch.einsum('ijl,ikl->ijk', (torch.matmul(h, self.A), h))
#         e = e + e.permute((0, 2, 1))  # e is already attention coefficient
#         e = e * (1 / math.sqrt(adj.shape[-1]))
#         zero_vec = -9e15 * torch.ones_like(e)
#         attention = torch.where(adj > 0, e, zero_vec)
#         attention = F.softmax(attention, dim=1)
#         attention = attention * adj
#         h_prime = F.relu(torch.einsum('aij,ajk->aik', (attention, h)))
#
#         coeff = torch.sigmoid(self.gate(torch.cat([x, h_prime], -1))).repeat(1, 1, x.size(-1))
#         retval = coeff * x + (1 - coeff) * h_prime
#         return retval


class GATgate_lp2(nn.Module):
    def __init__(self, n_out_feature):
        super(GATgate_lp2, self).__init__()
        self.w_l = nn.Linear(n_out_feature, n_out_feature)
        self.w_p = nn.Linear(n_out_feature, n_out_feature)
        self.LR = nn.LeakyReLU()

    def forward(self, vec_l, vec_p, adj_inter):# adj_l: max_num_atoms, max_num_atoms / adj_p: max_num_seq, max_num_atoms
        h_l = self.w_l(vec_l)
        h_p = self.w_p(vec_p)

        intermat = torch.einsum('aij,ajk->aik', (h_l, h_p.transpose(-1, -2)))# batch, max_num_atoms, max_num_seq
        intermat = intermat*adj_inter# batch, max_num_atoms, max_num_seq
        return intermat


class GATgate_lp(nn.Module):
    def __init__(self, n_out_feature):
        super(GATgate_lp, self).__init__()
        self.n_out = n_out_feature
        self.w_l = nn.Linear(n_out_feature, n_out_feature)
        self.w_p = nn.Linear(n_out_feature, n_out_feature)
        self.LR = nn.LeakyReLU()

    def forward(self, vec_l, vec_p, adj_inter):
        h_l = self.w_l(vec_l)
        h_p = self.w_p(vec_p)

        h_l2 = torch.einsum('aij,ajk->aik', (adj_inter, h_p))  # batch, num_atoms, embedding
        h_l2 = self.LR(h_l2*h_l)

        h_p2 = torch.einsum('aij,ajk->aik', (adj_inter.transpose(-1, -2), h_l))  # batch, num_seq, embedding
        h_p2 = self.LR(h_p2 * h_p)
        return h_l2, h_p2







