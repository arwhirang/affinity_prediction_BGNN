import math
import time
import pickle
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
# from torch.nn.utils import weight_norm
# from sru import SRU, SRUCell
from pdbbind_utils_bsp import *

# some predefined parameters
# elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K',
#              'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In',
#              'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U',
#              'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs', 'unknown']
# atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
# bond_fdim = 6
max_nb = 6
embedding_dim = 200
max_num_seq = 1000
max_num_atoms = 150
cudaDevid = 2
# define the model


class CustomRSum(nn.Module):
    def __init__(self):
        super(CustomRSum, self).__init__()

    def forward(self, inputs, dWhich):
        return torch.sum(inputs * dWhich, dim=1)  # only the 1 instnce survives


class Net(nn.Module):
    def __init__(self, atom_vocab_size):
        super(Net, self).__init__()

        """rak selfattn"""
        # self.num_layers = 3
        # self.enc_layers = [EncoderLayer(embedding_dim, 1, 1024, Drate=0.5).cuda(cudaDevid) for _ in range(self.num_layers)]

        self.dropoutG = 0.3
        self.embed_atom = nn.Embedding(atom_vocab_size, embedding_dim)
        # self.threeConv_lig = nn.Sequential(
        #     # Input: (N, C_{in}, H_{in}, W_{in})
        #     # Output: (N, C_{out}, H_{out}, W_{out})
        #     # n_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T, T]], stride: Union[T, Tuple[T, T]] = 1, padding: Union[T, Tuple[T, T]] = 0, dilation: Union[T, Tuple[T, T]] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'
        #     nn.Conv2d(1, 64, (max_num_atoms, 1), stride=(max_num_atoms, 1), padding=0, bias=False),
        #     nn.LeakyReLU()
        # )
        # self.threeConv_poc = nn.Sequential(
        #     # Input: (N, C_{in}, H_{in}, W_{in})
        #     # Output: (N, C_{out}, H_{out}, W_{out})
        #     # n_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T, T]], stride: Union[T, Tuple[T, T]] = 1, padding: Union[T, Tuple[T, T]] = 0, dilation: Union[T, Tuple[T, T]] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'
        #     nn.Conv2d(1, 64, (max_num_seq, 1), stride=(max_num_seq, 1), padding=0, bias=False),
        #     nn.LeakyReLU()
        # )
        self.threeConv = nn.Sequential(
            # Input: (N, C_{in}, H_{in}, W_{in})
            # Output: (N, C_{out}, H_{out}, W_{out})
            # n_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T, T]], stride: Union[T, Tuple[T, T]] = 1, padding: Union[T, Tuple[T, T]] = 0, dilation: Union[T, Tuple[T, T]] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'
            nn.Conv2d(1, 16, 3, 1, 1, bias=False),
            nn.MaxPool2d(3, stride=2, padding=1),  # divide the H_out and W_out by half
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.threeConv2 = nn.Sequential(
            # Input: (N, C_{in}, H_{in}, W_{in})
            # Output: (N, C_{out}, H_{out}, W_{out})
            # n_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T, T]], stride: Union[T, Tuple[T, T]] = 1, padding: Union[T, Tuple[T, T]] = 0, dilation: Union[T, Tuple[T, T]] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'
            nn.Conv2d(1, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )

        self.FC1_protein = nn.Linear(embedding_dim, embedding_dim)
        self.FC1_ligand = nn.Linear(embedding_dim, embedding_dim)
        self.FC2_CNN1 = nn.Linear(64 * 25 * 25, 128)

        self.FC3_Final1_kikd = nn.Linear(128, 32)
        self.FC3_Final2_kikd = nn.Linear(32, 1)
        self.FC4 = nn.Linear(2, 1)

        #self.FC3_Final1_ic50 = nn.Linear(embedding_dim*embedding_dim, embedding_dim)
        #self.FC3_Final2_ic50 = nn.Linear(embedding_dim, 1)
        self.dropoutF = nn.Dropout(0.3)

        """Pairwise Interaction Prediction Module"""
        self.FCpip_pairwise = nn.Linear(16, 1)
        self.sig = nn.Sigmoid()

        """Graph whatever"""
        self.N_atom_features = 34
        self.embede = nn.Linear(self.N_atom_features, 140, bias=False)
        self.bn = nn.BatchNorm1d(1150)
        self.FC = nn.ModuleList([nn.Linear(140, 64) if i == 0 else
                                 nn.Linear(64, 1) for i in range(2)])#if i == 1
        self.gconv1 = nn.ModuleList([GATgate(140, 140) for _ in range(3)])

    def pairwise_pred_module(self, batch_size, lig_feature, poc_feature, lig_mask, poc_mask):
        pairwise_ligand = lig_feature#self.LReLU_pred(lig_feature)# batch, max_num_atoms, embedding_dim
        pairwise_pocket = poc_feature#self.LReLU_pred(poc_feature)# batch, max_num_seq, embedding_dim
        pairwise_pred = torch.matmul(pairwise_ligand, pairwise_pocket.transpose(1, 2))# batch, max_num_atoms, max_num_seq
        pairwise_mask = torch.matmul(lig_mask.view(batch_size, -1, 1), poc_mask.view(batch_size, 1, -1))# batch, max_num_atoms, max_num_seq

        threeConvRes2 = self.threeConv2(pairwise_pred.view(-1, 1, max_num_atoms, max_num_seq))# batch, 16, max_num_atoms, max_num_seq
        threeConvRes2 = torch.transpose(threeConvRes2, 1, 2)
        threeConvRes2 = torch.transpose(threeConvRes2, 2, 3)
        pairwise_pred_fin = self.FCpip_pairwise(threeConvRes2)# batch, max_num_atoms, max_num_seq, 1
        pairwise_pred_fin = pairwise_pred_fin.view(-1, max_num_atoms, max_num_seq).contiguous() * pairwise_mask# batch, max_num_atoms, max_num_seq
        return pairwise_pred_fin

    def GraphConv_module(self, vertex_feature, atom_adj1, isTrain=False):
        vertex_feature = self.embede(vertex_feature)
        for k in range(3):
            vertex_tmp1 = self.gconv1[k](vertex_feature, atom_adj1)
            vertex_feature = vertex_tmp1
            vertex_feature = F.dropout(vertex_feature, p=self.dropoutG, training=isTrain)
        return vertex_feature# batch_size,  max_num_atoms, embedding_dim

    def graphFC(self, c_hs, isTrain=False):
        for k in range(len(self.FC)):
            if k < len(self.FC) - 1:
                c_hs = self.FC[k](c_hs)
                #c_hs = F.dropout(c_hs, p=self.dropoutF, training=isTrain)
            else:
                c_hs = self.FC[k](c_hs)
        c_hs = F.relu(c_hs)
        return c_hs

    def forward(self, L_mask, vecOfLatoms, P_mask, vecOfPatoms, indexForModel, isTrain=False):
        batch_size = vecOfLatoms.size(0)
        L_embeds = self.embed_atom(vecOfLatoms) # batch, max_num_atoms, embedding_dim
        LFC = self.FC1_ligand(L_embeds) # batch, max_num_atoms, embedding_dim
        #graph_feature, vertex_feature = self.GraphConv_module_new(batch_size, L_mask, vecOfLatoms, anb)
        P_embeds = self.embed_atom(vecOfPatoms)  # batch, max_num_seq, embedding_dim
        PFC = self.FC1_protein(P_embeds)  # batch, max_num_seq, embedding_dim

        pairwise_pred = self.pairwise_pred_module(batch_size, LFC, PFC, L_mask, P_mask)

        LtoP = torch.matmul(LFC.transpose(1, 2), pairwise_pred)  # batch, embedding_dim, max_num_seq
        LPMul = torch.matmul(LtoP, PFC)  # batch, embedding_dim, embedding_dim
        threeConvRes = self.threeConv(LPMul.view(-1, 1, embedding_dim, embedding_dim))  # batch, 64*25*25
        threeConvRes = torch.transpose(threeConvRes, 1, 2)
        threeConvRes = torch.transpose(threeConvRes, 2, 3)
        threeConvRes = self.FC2_CNN1(threeConvRes.contiguous().view(-1, 64 * 25 * 25))
        threeConvRes = self.dropoutF(threeConvRes)

        finStage = threeConvRes  # torch.cat(threeConvRes], -1)
        finStage_kikd = self.FC3_Final1_kikd(finStage)  # batch, 32
        finStage_kikd = self.FC3_Final2_kikd(finStage_kikd)  # batch, 1
        return finStage_kikd.view(-1), self.sig(pairwise_pred)


class GATgate(nn.Module):
    def __init__(self, n_in_feature, n_out_feature):
        super(GATgate, self).__init__()
        self.n_in = n_in_feature
        self.n_out = n_out_feature
        self.W = nn.Linear(n_out_feature, n_out_feature)
        self.A = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))
        self.gate = nn.Linear(n_out_feature * 2, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)


    def forward(self, x, adj):
        h = self.W(x)
        e = torch.einsum('ijl,ikl->ijk', (torch.matmul(h, self.A), h))
        e = e + e.permute((0, 2, 1))  # e is already attention coefficient
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = attention * adj
        h_prime = F.relu(torch.einsum('aij,ajk->aik', (attention, h)))

        coeff = self.gate(torch.cat([x, h_prime], -1)).repeat(1, 1, x.size(-1))
        retval = coeff * x + (1 - coeff) * h_prime
        return retval


