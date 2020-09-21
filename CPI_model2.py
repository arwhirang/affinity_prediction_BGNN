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
        self.dropout = nn.Dropout(0.3)

        #self.embed_protein = nn.Embedding(vocab_size, embedding_dim)
        #self.embed_ligand = nn.Embedding(vocab_size, embedding_dim)
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
        self.FC2_Final1 = nn.Linear(128, 32)
        self.FC2_Final2 = nn.Linear(32, 1)

        """Pairwise Interaction Prediction Module"""
        # self.FCpip_protein = nn.Linear(embedding_dim, embedding_dim)
        # self.FCpip_ligand = nn.Linear(embedding_dim, embedding_dim)
        self.FCpip_pairwise = nn.Linear(16, 1)
        self.FCpip_mulMat = nn.Linear(16, 1)
        #self.LReLU_pred = nn.LeakyReLU()
        self.sig = nn.Sigmoid()

        """Graph whatever"""
        self.label_U2 = nn.Linear(embedding_dim + max_nb, embedding_dim)
        self.label_U1 = nn.Linear(2*embedding_dim, embedding_dim)
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

        threeConvRes3 = self.threeConv3(pairwise_pred.view(-1, 1, max_num_atoms, max_num_seq))  # batch, 16, max_num_atoms, max_num_seq
        threeConvRes3 = torch.transpose(threeConvRes3, 1, 2)
        threeConvRes3 = torch.transpose(threeConvRes3, 2, 3)
        mulMat = self.FCpip_mulMat(threeConvRes3)  # batch, max_num_atoms, max_num_seq, 1
        mulMat = mulMat.view(-1, max_num_atoms, max_num_seq).contiguous() * pairwise_mask# batch, max_num_atoms, max_num_seq
        return pairwise_pred_fin, mulMat

    def wln_unit(self, batch_size, vertex_mask, vertex_features, edge_initial, atom_adj, bond_adj, nbs_mask):
        nbs_mask = nbs_mask.view(batch_size, max_num_atoms, max_nb, 1)
        vertex_nei = torch.index_select(vertex_features.view(-1, embedding_dim), 0, atom_adj).\
            view(batch_size,  max_num_atoms, max_nb, embedding_dim)
        edge_nei = torch.index_select(edge_initial.view(-1, max_nb), 0, bond_adj).\
            view(batch_size, max_num_atoms, max_nb, max_nb)
        l_nei = torch.cat((vertex_nei, edge_nei), -1)#batch_size,  max_num_atoms, max_nb, embedding_dim + max_nb
        nei_label = F.leaky_relu(self.label_U2(l_nei), 0.1)
        nei_label = torch.sum(nei_label * nbs_mask, dim=-2)#batch_size,  max_num_atoms, embedding_dim
        new_label = torch.cat((vertex_features, nei_label), 2)#batch_size,  max_num_atoms, 2*embedding_dim
        new_label = self.label_U1(new_label)#batch_size,  max_num_atoms, embedding_dim
        graph_feature = F.leaky_relu(new_label, 0.1)
        return graph_feature

    def GraphConv_module(self, batch_size, vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask):
        vertex_initial = self.embed_atom(vertex)  # batch, max_num_atoms, embedding_dim
        edge_initial = self.embed_bond(edge)  # batch, max_num_atoms, embedding_dim
        vertex_feature = F.leaky_relu(self.FC1_ligand(vertex_initial), 0.1)# batch, max_num_atoms, embedding_dim
        # batch_size,  max_num_atoms, embedding_dim
        graph_feature = self.wln_unit(batch_size, vertex_mask, vertex_feature, edge_initial, atom_adj, bond_adj, nbs_mask)
        return graph_feature, vertex_feature#, super_feature

    def forward(self, L_mask, vecOfLatoms, fb, anb, bnb, nbs_mat, vecOfPatoms, P_mask, tmpFRIMat):
        batch_size = vecOfLatoms.size(0)
        #L_embeds = self.embed_atom(vecOfLatoms) # batch, max_num_atoms, embedding_dim
        #LFC = self.FC1_ligand(L_embeds) # batch, max_num_atoms, embedding_dim
        graph_feature, vertex_feature = self.GraphConv_module(batch_size, L_mask, vecOfLatoms, fb, anb, bnb, nbs_mat)

        P_embeds = self.embed_atom(vecOfPatoms)  # batch, max_num_seq, embedding_dim
        PFC = self.FC1_protein(P_embeds)  # batch, max_num_seq, embedding_dim

        pairwise_pred, mulMat = self.pairwise_pred_module(batch_size, vertex_feature, PFC, L_mask, P_mask, tmpFRIMat)

        LtoP = torch.matmul(graph_feature.transpose(1, 2), mulMat)  # batch, embedding_dim, max_num_seq
        LPMul = torch.matmul(LtoP, PFC)  # batch, embedding_dim, embedding_dim
        threeConvRes = self.threeConv(LPMul.view(-1, 1, embedding_dim, embedding_dim))# batch, 64*25*25
        threeConvRes = torch.transpose(threeConvRes, 1, 2)
        threeConvRes = torch.transpose(threeConvRes, 2, 3)
        threeConvRes = self.FC2_CNN1(threeConvRes.contiguous().view(-1, 64*25*25))
        threeConvRes = self.dropout(threeConvRes)
        threeConvRes = self.FC2_Final1(threeConvRes)#batch, 32
        finStage = self.FC2_Final2(threeConvRes)#batch, 1
        return finStage.view(-1), self.sig(pairwise_pred)





