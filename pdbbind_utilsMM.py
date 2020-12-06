import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, precision_score, roc_auc_score
from scipy.spatial import distance_matrix
import time
import torch
from torch import nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
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

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    # 허용되지 않은 입력을 마지막 요소로 매핑
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# def one_of_k_encoding_unk(x, allowable_set):
#     """Maps inputs not in the allowable set to the last element."""
#     # 허용되지 않은 입력을 마지막 요소로 매핑
#     if x not in allowable_set:
#         x = allowable_set[-1]
#     return allowable_set.index(x)


def atom_feature(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']) +
                    one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) +
                    [atom.GetIsAromatic()])#34


# def atom_feature(atom):
#     return one_of_k_encoding_unk(atom.GetSymbol(), ['None', 'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']),\
#            one_of_k_encoding_unk(atom.GetDegree(), [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),\
#            one_of_k_encoding_unk(atom.GetTotalNumHs(), [-1, 0, 1, 2, 3, 4]),\
#            one_of_k_encoding_unk(atom.GetImplicitValence(), [-1, 0, 1, 2, 3, 4, 5, 6]),\
#            one_of_k_encoding_unk(atom.GetIsAromatic(), [-1, False, True])


def Mol2Graph(lig_mol, poc_mol, dicAtom2I):
    n_l_atoms = lig_mol.GetNumAtoms()
    n_p_atoms = poc_mol.GetNumAtoms()
    atoms_lig = lig_mol.GetAtoms()
    atoms_poc = poc_mol.GetAtoms()
    #adj_mat = np.zeros((n_l_atoms+n_p_atoms, n_l_atoms+n_p_atoms), dtype=np.int32)

    if n_p_atoms >= max_num_seq:
        print('PocketPatoms length is too long ' + str(n_p_atoms))
        return [], [], []
    if n_l_atoms >= max_num_atoms:
        print(' len(ligand) is too long ' + str(n_l_atoms) + ' ==> we disregard such instances')
        return [], [], []

    vec_lig1 = np.zeros([n_l_atoms, 34])
    vec_poc1 = np.zeros([n_p_atoms, 34])

    for i, ele in enumerate(atoms_lig):
        vec_lig1[i] = atom_feature(ele)
    # adj1 = GetAdjacencyMatrix(lig_mol) + np.eye(n_l_atoms)
    #adj_mat[:n_l_atoms, :n_l_atoms] = adj1

    for i, ele in enumerate(atoms_poc):
        vec_poc1[i] = atom_feature(ele)
    #adj2 = GetAdjacencyMatrix(poc_mol) + np.eye(n_p_atoms)
    #adj_mat[n_l_atoms:n_l_atoms+n_p_atoms, n_l_atoms:n_l_atoms+n_p_atoms] = adj2

    c1 = lig_mol.GetConformers()[0]
    c2 = poc_mol.GetConformers()[0]
    adj_inter = distance_matrix(np.array(c1.GetPositions()), np.array(c2.GetPositions()))
    thres = makeThreshold()
    cutoffDist = 6.95  # angstroms
    for i in range(n_l_atoms):
        for j in range(n_p_atoms):
            # ligtype, ligPos, poctype, pocPos
            eucDist = adj_inter[i][j]
            if eucDist > cutoffDist:
                currVal = 0
            else:
                currVal = cutoffBinning(eucDist, thres)
                #currVal = fri(atoms_lig[i], atoms_poc[j], eucDist)
            adj_inter[i][j] = currVal

    return vec_lig1, vec_poc1, adj_inter


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


def get_mask(arr_list, N):
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        n = len(arr)
        for j in range(n):
            if arr_list[i][j] != 0:
                a[i][j] = 1
    return a


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
    atom_serials_poc, atom_ele_list_poc, atom_pos_poc = [], [], []
    f = open(pdbName, "r")
    lines = f.readlines()
    for i, aline in enumerate(lines):
        if not aline.startswith("ATOM"):
            continue

        splitted = aline.strip().split()
        xpos, ypos, zpos = get_pos_iter(splitted)
        atom_pos_poc.append((xpos, ypos, zpos))
        atomSerialNum = splitted[1]
        atom_serials_poc.append(leave_only_digits(atomSerialNum))
        atomtype = get_atomtype(splitted)
        atom_ele_list_poc.append(atomtype)

    f.close()
    if atom_serials_poc:
        return atom_serials_poc, atom_ele_list_poc, atom_pos_poc
    else:
        return None, None, None


def euclidDist(ligtype, ligPos, poctype, pocPos):
    # ligtypeList = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
    # poctypeList = ["C", "N", "O", "S"]

    ligx, ligy, ligz = ligPos
    pocx, pocy, pocz = pocPos
    # if ligtype not in ligtypeList:
    #     return 99999
    # if poctype not in poctypeList:
    #     return 99999
    if ligtype == "H" or poctype == "H" or poctype == "D" or poctype == "X" or poctype == "XE":
        return 99999

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


def fri(ligAtom, pocAtom, eucDist):
    ligRadi = Chem.GetPeriodicTable().GetRvdw(ligAtom.GetSymbol())
    pocRadi = Chem.GetPeriodicTable().GetRvdw(pocAtom.GetSymbol())
    tau = 1
    evee = 5
    return 1 / (1 + math.pow(eucDist / (tau * (ligRadi + pocRadi)), evee))


def listOfCASF2016():
    retList = []
    f = open('../../data/aff_data_TestForCoreset2016.csv')
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


def get_ligand_from_pdb(ligand, pdbPath):
    # ligand side
    atom_serials = []
    atom_ele_list = []
    atom_pos = []
    f = open(pdbPath, "r")
    lines = f.readlines()

    for i, aline in enumerate(lines):
        if aline.startswith("ANISOU"):
            continue

        if aline.startswith("HETATM"):
            splitted = aline.strip().split()

            res = splitted[3]
            if len(res) != 3 or res != ligand:
                continue

            xpos, ypos, zpos = get_pos_iter(splitted)
            atom_pos.append((xpos, ypos, zpos))
            atomSerialNum = splitted[1]
            atom_serials.append(leave_only_digits(atomSerialNum))
            atomtype = get_atomtype(splitted)
            atom_ele_list.append(atomtype)
        else:
            continue
    f.close()

    if atom_serials:
        return atom_serials, atom_ele_list, atom_pos
    else:
        return None, None, None


def load_data2(dicAtom2I):
    datapack_test = []
    datapack_kikd = []
    datapack_ic50 = []

    casfList = listOfCASF2016()
    f = open('../../pdbbind_index/INDEX_all.2019')
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
        # if value <= 2 or value >= 12:
        #     print('value is outside of typical range')
        #     continue
        if not os.path.exists("../../pdbbind_files/" + pdbid):  # some pdbid only exists in the index files
            continue

        sdfMOLs = Chem.rdmolfiles.MolFromMol2File("../../pdbbind_files/" + pdbid + "/" + pdbid + "_ligand.mol2",
                                                  sanitize=True, removeHs=False)
        if not sdfMOLs:
            sdfMOLs = Chem.SDMolSupplier("../../pdbbind_files/" + pdbid + "/" + pdbid + "_ligand.sdf")[0]
            if not sdfMOLs:
                print("not a valid mol-ligand1")
                continue

        pdbpoc_mol = Chem.rdmolfiles.MolFromPDBFile("../../pdbbind_files/" + pdbid + "/" + pdbid + "_pocket.pdb",
                                                    sanitize=True, removeHs=False)
        if not pdbpoc_mol:
            print("not a valid poc-mol1")
            continue

        vec_lig1, vec_poc1, adj_inter = Mol2Graph(sdfMOLs, pdbpoc_mol, dicAtom2I)
        if len(vec_lig1) == 0:
            print("not a valid lig-mol2")
            continue
        if len(vec_poc1) == 0:# or charge_poc is None:
            print("not a valid poc-mol2")
            continue

        # either KIKD or IC50
        if measure in ['Ki', 'Kd']:
            if pdbid in casfList:
                datapack_test.append([vec_lig1, adj_inter, vec_poc1, value, pdbid])
            else:
                datapack_kikd.append([vec_lig1, adj_inter, vec_poc1, value, pdbid])
        elif measure == "IC50":
            datapack_ic50.append([vec_lig1, adj_inter, vec_poc1, value, pdbid])

    f.close()
    return datapack_kikd, datapack_ic50, datapack_test


# Model parameter intializer
def weights_init(m):
    for param in m.parameters():
        if param.dim() == 1:
            continue
        else:
            nn.init.xavier_normal_(param)
