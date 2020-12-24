import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, precision_score, roc_auc_score
from scipy.spatial import distance_matrix
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from rdkit import Chem
# from rdkit.Chem.rdmolops import GetAdjacencyMatrix
# from rdkit.Chem import rdPartialCharges
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import math
from metrics import *
import os

max_num_seq = 2000
max_num_atoms = 200

twoDic = {"AL": "Al", "AU": "Au", "AG": "Ag", "AS": "As", "BA": "Ba", "BE": "Be", "BI": "Bi", "BR": "Br",
          "CA": "Ca", "CD": "Cd", "CL": "Cl", "CO": "Co", "CR": "Cr", "CS": "Cs", "CU": "Cu", "DY": "Dy",
          "EU": "Eu", "FE": "Fe", "GA": "Ga", "GD": "Gd", "GE": "Ge", "HG": "Hg", "IN": "In", "IR": 'Ir',
          "LI": "Li", "MG": "Mg", "MN": "Mn", "MO": "Mo", "NA": "Na", "ND": "Nd", "NI": "Ni", "PB": "Pb",
          "OS": "Os", "PD": "Pd", "PT": "Pt", "RB": "Rb", "RE": "Re", "RH": "Rh", "RU": "Ru", "SB": "Sb",
          "SE": "Se", "SI": "Si", "SN": "Sn", "SR": "Sr", "TI": "Ti", "TL": "Tl", "YB": "Yb", "ZN": "Zn",
          "ZR": "Zr"}


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        return str(len(allowable_set) - 1)
    return str(allowable_set.index(x))


def atom_features(atom):
    elem_list = ['<PAD>', 'B', 'C', 'F', 'H', 'I', 'K', 'N', 'O', 'P', 'S', 'U', 'V', 'W', 'Ag', 'Al', 'As', 'Au', 'Ba',
                 'Be', 'Bi', 'Br', 'Ca', 'Cd', 'Ce', 'Cl', 'Co', 'Cr', 'Cs', 'Cu', 'Dy', 'Eu', 'Fe', 'Ga', 'Gd', 'Ge',
                 'Hf', 'Hg', 'In', 'Ir', 'Li', 'Mg', 'Mn', 'Mo', 'Na', 'Nb', 'Ni', "Nd", "Pb", "Pd", "Pt", "Rb", 'Os',
                 'Re', 'Rh', "Ru", "Sb", "Se", "Si", 'Sm', "Sn", "Sr", 'Tc', 'Te', "Ti", "Tl", "Yb", "Zn", "Zr",
                 'unknown']
    return onek_encoding_unk(atom, elem_list)


# def mol2graph(lig_mol, list_atypes):
#     n_l_atoms = lig_mol.GetNumAtoms()
#     atoms = lig_mol.GetAtoms()
#
#     if len(atoms) != len(list_atypes):
#         return []
#     for i, atom in enumerate(atoms):
#         if list_atypes[i] != atom.GetSymbol():
#             return []
#
#     adj_mat = np.zeros((max_num_atoms+max_num_seq, max_num_atoms+max_num_seq), dtype=np.int32)
#
#     adj1 = GetAdjacencyMatrix(lig_mol) + np.eye(n_l_atoms)
#     adj_mat[:n_l_atoms, :n_l_atoms] = adj1
#     return adj_mat


# def mol_add_charge(cur_mol, max_seq):
#     charges = np.zeros((max_seq), dtype=np.int32)
#
#     #rdPartialCharges.ComputeGasteigerCharges(cur_mol)
#     AllChem.ComputeGasteigerCharges(cur_mol)
#     atoms = cur_mol.GetAtoms()
#     for i, atom in enumerate(atoms):
#         print(atom.GetSymbol(), atom.GetProp('_GasteigerCharge'))#too frequent None occurred
#         charges[i] = float(atom.GetProp('_GasteigerCharge'))
#     return charges


def reg_scores(label, pred):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    assert len(pred) == len(label)
    return rmse(label, pred), pearson(label, pred), spearman(label, pred)


def get_pos_iter(splitted):
    xposIndex = -6
    yposIndex = -5
    zposIndex = -4
    if len(splitted[-2]) > 6:  # no space between
        xposIndex = xposIndex + 1
        yposIndex = yposIndex + 1
        zposIndex = zposIndex + 1

    if len(splitted[zposIndex]) > 7:  # no space between z and y
        if len(splitted[zposIndex]) > 15:  # no space between z and y and x
            zpos = float(splitted[zposIndex][-8:])
            ypos = float(splitted[zposIndex][-16:-8])
            xpos = float(splitted[zposIndex][:-16])
        else:
            xposIndex = xposIndex + 1
            zpos = float(splitted[zposIndex][-8:])
            ypos = float(splitted[zposIndex][:-8])
            xpos = float(splitted[xposIndex])
    else:
        zpos = float(splitted[zposIndex])
        if len(splitted[yposIndex]) > 7:  # no space between y and x
            ypos = float(splitted[yposIndex][-8:])
            xpos = float(splitted[yposIndex][:-8])
        else:
            ypos = float(splitted[yposIndex])
            xpos = float(splitted[xposIndex])
    return xpos, ypos, zpos


def get_atomtype(splitted):
    atomtype = splitted[-1]
    tmplst = []
    for char in atomtype:
        if char.isalpha():
            tmplst.append(char)
        else:
            break
    atomtype = "".join(tmplst)
    if atomtype in twoDic:
        atomtype = twoDic[atomtype]
    return atomtype


# pocket pdb
# returns position dict and list of atoms
# _pocket.pdb and the original pdb file have not many things in common ==> position is common ==> key:position
def get_pocket_from_pdb(pdbName):
    atom_serials_poc, atom_ele_poc, atom_pos_poc = [], [], []
    f = open(pdbName, "r")
    lines = f.readlines()
    for i, aline in enumerate(lines):
        if not aline.startswith("ATOM"):
            continue

        splitted = aline.strip().split()
        atomtype = get_atomtype(splitted)
        if atomtype == "D":
            continue
        atom_ele_poc.append(atomtype)
        xpos, ypos, zpos = get_pos_iter(splitted)
        atom_pos_poc.append((xpos, ypos, zpos))
        atom_serial_num = splitted[1]
        atom_serials_poc.append(leave_only_digits(atom_serial_num))
    f.close()
    if atom_serials_poc:
        return atom_serials_poc, atom_ele_poc, atom_pos_poc
    else:
        return None, None, None


def euclidDist(ligPos, pocPos):
    # ligtypeList = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
    # poctypeList = ["C", "N", "O", "S"]

    ligx, ligy, ligz = ligPos
    pocx, pocy, pocz = pocPos
    # if ligtype not in ligtypeList:
    #     return 99999
    # if poctype not in poctypeList:
    #     return 99999

    xdist = math.pow(pocx - ligx, 2)
    ydist = math.pow(pocy - ligy, 2)
    zdist = math.pow(pocz - ligz, 2)
    return math.sqrt(xdist + ydist + zdist)


def cutoffBinning(eucDist, thres):#thres starts from 6.95 and ends at 2.75
    for i, ele in enumerate(thres):
        if eucDist > ele:
            return i/70.0
    return 1


def makeThreshold():
    thres = list(np.linspace(2.75, 6.95, 70))#usually 0.06 range for each bin
    for i, ele in enumerate(thres):
        thres[i] = math.floor(ele * 100) / 100
    thres.reverse()#starts from 6.95 and ends at 2.75
    return thres

"""
def fri(ligAtom, pocAtom, eucDist):
    # some atoms are like "O1-"...
    if any(chr.isdigit() for chr in pocAtom):
        pocAtom = "".join([chr for chr in pocAtom if chr.isalpha()])

    if ligAtom in twoDic:
        ligAtom = twoDic[ligAtom]
    if pocAtom in twoDic:
        pocAtom = twoDic[pocAtom]

    ligRadi = Chem.GetPeriodicTable().GetRvdw(ligAtom)
    pocRadi = Chem.GetPeriodicTable().GetRvdw(pocAtom)
    tau = 1
    evee = 15
    return 1 / (1 + math.pow(eucDist / (tau * (ligRadi + pocRadi)), evee))
"""

def listOfCASF2016():
    retList = []
    f = open('../data/aff_data_TestForCoreset2016.csv')
    for i, line in enumerate(f.readlines()):
        if i == 0:
            continue
        splitted = line.split(",")
        if splitted[-1].startswith("test"):
            retList.append(splitted[0])
    return retList


def leave_only_digits(astr):
    tmp = []
    for achar in astr:
        if achar.isdigit():
            tmp.append(achar)
    return int("".join(tmp))


# ligand mol2
def readmol2(mol2Name):
    poslist = []
    atypelist = []
    flag1 = False
    fmol2 = open(mol2Name, "r")
    lines = fmol2.readlines()
    for i, aline in enumerate(lines):
        if aline.startswith("@<TRIPOS>ATOM"):
            flag1 = True
        if aline.startswith("@<TRIPOS>BOND"):
            flag1 = False

        if not flag1:
            continue
        splitted = aline.strip().split()

        if len(splitted) != 9:
            continue
        atomtype = splitted[5].split(".")[0]
        if atomtype == "D":
            continue
        atypelist.append(atomtype)
        xpos = float(splitted[2])
        ypos = float(splitted[3])
        zpos = float(splitted[4])
        poslist.append((xpos, ypos, zpos))

    fmol2.close()
    return poslist, atypelist


# def get_ligand_from_pdb(ligand, pdbPath):
#     # ligand side
#     atom_serials = []
#     atom_ele_list = []
#     atom_pos = []
#     f = open(pdbPath, "r")
#     lines = f.readlines()
#     for i, aline in enumerate(lines):
#         if not aline.startswith("HETATM"):
#             continue
#         splitted = aline.strip().split()
#
#         if len(splitted[0]) > 6:  # HETATM12344 etc...
#             if len(splitted[1]) > 4: # C23AFRA etc...
#                 res = splitted[1][-3:]
#             else:
#                 res = splitted[2]
#             if len(res) != 3 or res != ligand:
#                 continue
#
#             atom_serial_num = splitted[0][6:]
#             atom_serials.append(leave_only_digits(atom_serial_num))
#
#         else:
#             if len(splitted[2]) > 4: # C23AFRA etc...
#                 res = splitted[2][-3:]
#             else:
#                 res = splitted[3]
#             if len(res) != 3 or res != ligand:
#                 continue
#             atom_serial_num = splitted[1]
#             atom_serials.append(leave_only_digits(atom_serial_num))
#
#         xpos, ypos, zpos = get_pos_iter(splitted)
#         atom_pos.append((xpos, ypos, zpos))
#         atomtype = get_atomtype(splitted)
#         atom_ele_list.append(atomtype)
#     f.close()
#
#     if atom_serials:
#         return atom_serials, atom_ele_list, atom_pos
#     else:
#         return None, None, None


def load_data2(dicAtom2I):
    datapack_test = []
    datapack_kikd = []
    datapack_ic50 = []

    casfList = listOfCASF2016()
    f = open('../pdbbind_index/INDEX_all.2019')
    for line in f.readlines():
        if line[0] == '#':
            continue

        ligand = line.strip().split('(')[1].split(')')[0]
        # errors
        if '-mer' in ligand:
            continue
        elif len(ligand) != 3:
            continue

        lines = line.split('/')[0].strip().split('  ')
        pdbid = lines[0]
        if '~' in lines[3]:
            continue
        elif '<' in lines[3]:
            measure = lines[3].split('<')[0]
            unit = lines[3].split('<')[1][-2:]
            if '=' in lines[3]:
                value = float(lines[3].split('<')[1][1:-2])
            else:
                value = float(lines[3].split('<')[1][:-2])
        elif '>' in lines[3]:
            measure = lines[3].split('>')[0]
            if '=' in lines[3]:
                value = float(lines[3].split('>')[1][1:-2])
            else:
                value = float(lines[3].split('>')[1][:-2])
            unit = lines[3].split('>')[1][-2:]
        else:
            measure = lines[3].split('=')[0]
            value = float(lines[3].split('=')[1][:-2])
            unit = lines[3].split('=')[1][-2:]

        if unit == 'nM':
            pvalue = -np.log10(value) + 9
        elif unit == 'uM':
            pvalue = -np.log10(value) + 6
        elif unit == 'mM':
            pvalue = -np.log10(value) + 3
        elif unit == 'pM':
            pvalue = -np.log10(value) + 12
        elif unit == 'fM':
            pvalue = -np.log10(value) + 15

        # get labels
        value = float(pvalue)
        """
        if value <= 2 or value >= 12:
            print('value is outside of typical range')
            continue
        if not os.path.exists("../pdbbind_files/" + pdbid):  # some pdbid only exists in the index files
            continue
        if not os.path.exists("../pdb_files/" + pdbid + ".pdb"):  # some pdbid only exists in the index files
            continue
        """
        # read data from pdbbind files
        # these lines are required since pdbbind has only 1 ligand and 1 pocket data even if there are 3 interactions
        # get key:positions in str, value:lig index dict from the .sdf
        # atom_serials, atom_ele_list, atom_pos = get_ligand_from_pdb(ligand, "../pdbbind_files/" + pdbid + "/" + pdbid +
        #                                                             "_ligand2.pdb")
        # if not atom_serials:
        #     print("weird ligand2 pdb", pdbid)
        #     continue
        atom_pos_lig, atom_ele_lig = readmol2("../pdbbind_files/" + pdbid + "/" + pdbid + "_ligand.mol2")
        atom_serials_poc, atom_ele_poc, atom_pos_poc = get_pocket_from_pdb("../pdbbind_files/" + pdbid + "/" + pdbid + "_pocket.pdb")

        n_latom = len(atom_ele_lig)
        n_patom = len(atom_ele_poc)

        if not atom_serials_poc:
            print("weird pocket pdb", pdbid)
            continue
        if n_patom >= max_num_seq or n_latom >= max_num_atoms:
            print('PocketPatoms length is too long ' + str(n_patom) + ' or len(ligand) is too long ' +
                  str( n_latom ) + ' ==> we disregard such instances')
            continue

        vec_latoms = np.zeros([n_latom])
        for i, ltype in enumerate(atom_ele_lig):
            vec_latoms[i] = dicAtom2I[atom_features(ltype)]
        vec_patoms = np.zeros([n_patom])
        for i, ptype in enumerate(atom_ele_poc):
            vec_patoms[i] = dicAtom2I[atom_features(ptype)]

        thres = makeThreshold()
        cutoffDist = 6.95  # angstroms
        adj_mat = np.zeros((max_num_atoms, max_num_seq))
        for i in range(n_latom):
            for j in range(n_patom):
                eucDist = euclidDist(atom_ele_poc[j], atom_pos_poc[j])
                if eucDist > cutoffDist:
                    currVal = 0
                else:
                    currVal = cutoffBinning(eucDist, thres)
                    #currVal = fri(atom_ele_lig[i], atom_ele_poc[j], eucDist)
                adj_mat[i][j] = currVal)

        # either KIKD or IC50
        if measure in ['Ki', 'Kd']:
            if pdbid in casfList:
                datapack_test.append([vec_latoms, adj_mat, vec_patoms, value, pdbid])
            else:
                datapack_kikd.append([vec_latoms, adj_mat, vec_patoms, value, pdbid])
        elif measure == "IC50":
            datapack_ic50.append([vec_latoms, adj_mat, vec_patoms, value, pdbid])

    f.close()
    return datapack_kikd, datapack_ic50, datapack_test


# Model parameter intializer
def weights_init(m):
    for param in m.parameters():
        if param.dim() == 1:
            continue
        else:
            nn.init.xavier_normal_(param)

