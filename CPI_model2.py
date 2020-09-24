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
from pdbbind_utils2 import *
from MultiHeadAttention import *

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
class Net(nn.Module):
    def __init__(self, atom_vocab_size, bond_vocab_size):
        super(Net, self).__init__()

        """rak selfattn"""
        # self.num_layers = 7
        # self.enc_layers = [EncoderLayer(self.hidden_size2, 1, 1024, [self.hidden_size2], Drate=0.1).cuda() for _ in
        #                    range(self.num_layers)]
        # selfattn_rak
        self.dropout = nn.Dropout(0.1)

        self.embed_atom = nn.Embedding(atom_vocab_size, embedding_dim)
        self.embed_bond = nn.Embedding(bond_vocab_size, max_nb)

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

        self.threeConv3 = nn.Sequential(
            # Input: (N, C_{in}, H_{in}, W_{in})
            # Output: (N, C_{out}, H_{out}, W_{out})
            # n_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T, T]], stride: Union[T, Tuple[T, T]] = 1, padding: Union[T, Tuple[T, T]] = 0, dilation: Union[T, Tuple[T, T]] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'
            nn.Conv2d(1, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )

        self.FC1_protein = nn.Linear(embedding_dim, embedding_dim)
        self.FC1_ligand = nn.Linear(embedding_dim, embedding_dim)

        self.FC2_CNN1 = nn.Linear(64*25*25, 128)
        self.FC2_Final1 = nn.Linear(128, 128)
        self.FC2_Final2 = nn.Linear(128, 1)

        """Pairwise Interaction Prediction Module"""
        self.FCpip_pairwise = nn.Linear(16, 1)
        self.FCpip_mulMat = nn.Linear(16, 1)
        self.sig = nn.Sigmoid()

        """Graph whatever"""
        self.gconv1 = nn.ModuleList([GATgate(embedding_dim, embedding_dim) for i in range(3)])
        self.LM = nn.LayerNorm(embedding_dim)
        # self.FCpip_merger = nn.Linear(max_num_seq*max_num_atoms, 128)
        # self.bn1d = nn.BatchNorm1d(128)

    def pairwise_pred_module(self, batch_size, lig_feature, poc_feature, lig_mask, poc_mask, tmpFRIMat):
        pairwise_ligand = lig_feature#self.LReLU_pred(lig_feature)# batch, max_num_atoms, embedding_dim
        pairwise_pocket = poc_feature#self.LReLU_pred(poc_feature)# batch, max_num_seq, embedding_dim
        pairwise_pred = torch.matmul(pairwise_ligand, pairwise_pocket.transpose(1, 2))# batch, max_num_atoms, max_num_seq
        pairwise_pred = pairwise_pred*tmpFRIMat
        pairwise_mask = torch.matmul(lig_mask.view(batch_size, -1, 1), poc_mask.view(batch_size, 1, -1))# batch, max_num_atoms, max_num_seq

        threeConvRes2 = self.threeConv2(pairwise_pred.view(-1, 1, max_num_atoms, max_num_seq))# batch, 16, max_num_atoms, max_num_seq
        threeConvRes2 = torch.transpose(threeConvRes2, 1, 2)
        threeConvRes2 = torch.transpose(threeConvRes2, 2, 3)
        pairwise_pred_fin = self.FCpip_pairwise(threeConvRes2)# batch, max_num_atoms, max_num_seq, 1
        pairwise_pred_fin = pairwise_pred_fin.view(-1, max_num_atoms, max_num_seq).contiguous() * pairwise_mask# batch, max_num_atoms, max_num_seq
        return pairwise_pred_fin#, mulMat

    def GraphConv_module_new(self, batch_size, vertex_mask, vertex, atom_adj):
        vertex_initial = self.embed_atom(vertex)  # batch, max_num_atoms, embedding_dim
        vertex_feature = F.leaky_relu(self.FC1_ligand(vertex_initial), 0.1)# batch, max_num_atoms, embedding_dim

        graph_feature = vertex_feature
        for k in range(1):
            graph_feature = self.gconv1[k](graph_feature, atom_adj)
            graph_feature = self.LM(graph_feature)
            #vertex_feature = self.dropout(vertex_feature)
        graph_feature = graph_feature * vertex_mask.view(batch_size, -1, 1)
        return graph_feature, vertex_feature# batch_size,  max_num_atoms, embedding_dim

    def forward(self, L_mask, vecOfLatoms, fb, anb, bnb, nbs_mat, vecOfPatoms, P_mask, tmpFRIMat):
        batch_size = vecOfLatoms.size(0)
        #L_embeds = self.embed_atom(vecOfLatoms) # batch, max_num_atoms, embedding_dim
        #LFC = self.FC1_ligand(L_embeds) # batch, max_num_atoms, embedding_dim
        graph_feature, vertex_feature = self.GraphConv_module_new(batch_size, L_mask, vecOfLatoms, anb)
        P_embeds = self.embed_atom(vecOfPatoms)  # batch, max_num_seq, embedding_dim
        PFC = self.FC1_protein(P_embeds)  # batch, max_num_seq, embedding_dim

        pairwise_pred = self.pairwise_pred_module(batch_size, vertex_feature, PFC, L_mask, P_mask, tmpFRIMat)#mulMat

        LtoP = torch.matmul(graph_feature.transpose(1, 2), tmpFRIMat)  # batch, embedding_dim, max_num_seq
        LPMul = torch.matmul(LtoP, PFC)  # batch, embedding_dim, embedding_dim
        threeConvRes = self.threeConv(LPMul.view(-1, 1, embedding_dim, embedding_dim))# batch, 64*25*25
        threeConvRes = torch.transpose(threeConvRes, 1, 2)
        threeConvRes = torch.transpose(threeConvRes, 2, 3)
        threeConvRes = self.FC2_CNN1(threeConvRes.contiguous().view(-1, 64*25*25))
        threeConvRes = self.dropout(threeConvRes)

        finStage = threeConvRes#torch.cat(threeConvRes], -1)
        finStage = self.FC2_Final1(finStage)#batch, 32
        finStage = self.FC2_Final2(finStage)#batch, 1
        return finStage.view(-1), self.sig(pairwise_pred)


class GATgate(torch.nn.Module):
    def __init__(self, n_in_feature, n_out_feature):
        super(GATgate, self).__init__()
        self.W = nn.Linear(n_in_feature, n_out_feature)
        self.A = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))
        self.gate = nn.Linear(n_out_feature * 2, n_out_feature)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x, adj):
        batch_size = x.size(0)
        h = self.W(x)
        e = torch.einsum('ijl,ikl->ijk', (torch.matmul(h, self.A), h))
        e = e + e.permute((0, 2, 1))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = attention * adj
        h_prime = F.relu(torch.einsum('aij,ajk->aik', (attention, h)))
        coeff = self.gate(torch.cat([x, h_prime], -1)).view(batch_size, -1, embedding_dim)#.repeat(1, 1, embedding_dim)
        retval = coeff * x + (1 - coeff) * h_prime
        return retval


