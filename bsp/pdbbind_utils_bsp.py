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
import math
from metrics import *
import os

# atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
# bond_fdim = 6
max_num_bonds = 100
max_nb = 6
max_num_seq = 1000
max_num_atoms = 150

# # embedding where function requires 2-D indicies
# def adjMat2D(input_array):
#     batch_size = np.shape(input_array)[0]
#     new_array = np.zeros((batch_size, max_num_atoms, max_num_atoms), dtype=np.int32)
#
#     for k in range(batch_size):
#         for i in range(max_num_atoms):
#             for j in range(max_nb):
#                 if input_array[k, i, j] != -1:
#                     new_array[k, i, input_array[i, j]] = 1
#     return new_array


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


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features_mol(atom):
    elem_list = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(), elem_list) +
                    one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) +
                    [atom.GetIsAromatic()])


def Mol2Graph(lig_mol, poc_mol):
    n_l_atoms = lig_mol.GetNumAtoms()
    n_p_atoms = poc_mol.GetNumAtoms()
    if n_l_atoms > max_num_atoms or n_p_atoms > max_num_seq:
        return [], [], 0

    vec_atoms = np.zeros(([max_num_atoms+max_num_seq, 34]), dtype=np.int32)  # atom feature ID
    adj_mat = np.zeros((max_num_atoms+max_num_seq, max_num_atoms+max_num_seq), dtype=np.int32)

    for atom in lig_mol.GetAtoms():
        idx = atom.GetIdx()
        vec_atoms[idx] = atom_features_mol(atom)
    for atom in poc_mol.GetAtoms():
        idx = atom.GetIdx()
        vec_atoms[idx + n_l_atoms] = atom_features_mol(atom)

    adj1 = GetAdjacencyMatrix(lig_mol) + np.eye(n_l_atoms)
    adj_mat[:n_l_atoms, :n_l_atoms] = adj1
    adj2 = GetAdjacencyMatrix(poc_mol) + np.eye(n_p_atoms)
    adj_mat[n_l_atoms:n_l_atoms+n_p_atoms, n_l_atoms:n_l_atoms+n_p_atoms] = adj2

    # adj_mat2 = np.copy(adj_mat)
    # c1 = lig_mol.GetConformers()[0]  # GetConformers( (Mol)arg1) -> object : Get all the conformers as a tuple
    # d1 = np.array(c1.GetPositions())
    # c2 = poc_mol.GetConformers()[0]  # GetConformers( (Mol)arg1) -> object : Get all the conformers as a tuple
    # d2 = np.array(c2.GetPositions())
    # dm = distance_matrix(d1, d2)
    # adj_mat2[:n_l_atoms, n_l_atoms:] = np.copy(dm)
    # adj_mat2[n_l_atoms:, :n_l_atoms] = np.copy(np.transpose(dm))

    return vec_atoms, adj_mat, n_l_atoms


def reg_scores(label, pred):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    assert len(pred) == len(label)
    return rmse(label, pred), pearson(label, pred), spearman(label, pred)


# use plip_results
def get_interacts(pdbid, cid, ori_ligand_dic, ori_pocket_dic, pairwise_mat):
    # print(pdbid, cid)
    pairwise_exist = 0
    # bond_list = []
    f = open('../plip_results/' + pdbid + 'out.txt')
    isheader = False
    for line in f.readlines():
        if line[0] == '*':
            bond_type = line.strip().replace('*', '')
            isheader = True
        if line[0] == '|':
            if isheader:
                header = line.replace(' ', '').split('|')
                isheader = False
                continue
            lines = line.replace(' ', '').split('|')
            if cid not in lines[5]:
                continue
            # aa_id, aa_name, aa_chain, ligand_id, ligand_name, ligand_chain = int(lines[1]), lines[2], lines[3], int(lines[4]), lines[5], lines[6]
            if bond_type in ['Hydrogen Bonds', 'Water Bridges']:
                atom_idx1, atom_idx2 = lines[12], lines[14]
                if atom_idx1 in ori_ligand_dic and atom_idx2 in ori_ligand_dic:  # discard ligand-ligand interaction
                    continue
                if atom_idx1 in ori_ligand_dic:
                    atom_idx_ligand, atom_idx_protein = atom_idx1, atom_idx2
                elif atom_idx2 in ori_ligand_dic:
                    atom_idx_ligand, atom_idx_protein = atom_idx2, atom_idx1
                else:
                    continue
                pairwise_mat[ori_ligand_dic[atom_idx_ligand]][ori_pocket_dic[atom_idx_protein]] = 1
                pairwise_exist = 1
            elif bond_type == 'Hydrophobic Interactions':
                continue
            elif bond_type in ['pi-Stacking', 'pi-Cation Interactions']:
                continue
            elif bond_type == 'Salt Bridges':
                continue
            elif bond_type == 'Halogen Bonds':
                continue
            elif bond_type == 'Metal Complexes':
                continue  # maybe later
            else:
                print('weird bond_type:', bond_type)
                print(header)
                print(lines)
    f.close()
    return pairwise_exist


def get_mask(arr_list, N):
    a = np.zeros((len(arr_list), N))
    # isAlreadyZero = False
    for i, arr in enumerate(arr_list):
        n = len(arr)
        for j in range(n):
            if arr_list[i][j] != 0:
                a[i][j] = 1
    return a


# returns position dict and list of atoms
def readsdf(sdfName):
    dicPOS_sdf = {}
    listOfAtypes = []
    ligPOS_Index = {}  # key:positions in str, value:lig index in the .sdf
    fsdf = open(sdfName, "r")
    lines = fsdf.readlines()
    indexForLigVal = 0
    for i, aline in enumerate(lines):
        if i < 4:
            continue
        splitted = aline.strip().split()
        if len(splitted) != 10:
            break
        atomtype = splitted[3]
        if atomtype in twoDic:
            atomtype = twoDic[atomtype]
        # if atomtype == "H":
        #     continue
        xpos = float(splitted[0])
        ypos = float(splitted[1])
        zpos = float(splitted[2])

        dicPOS_sdf[indexForLigVal] = (atomtype, xpos, ypos, zpos)
        listOfAtypes.append(atomtype)
        assert atomtype != "D"
        ligPOS_Index["_".join([atomtype, str(xpos), str(ypos), str(zpos)])] = indexForLigVal
        indexForLigVal += 1
    fsdf.close()
    return dicPOS_sdf, listOfAtypes, ligPOS_Index


# returns position dict and list of atoms
def readmol2(mol2Name):
    dicPOS_mol2 = {}
    listOfAtypes = []
    ligPOS_Index = {}  # key:positions in str, value:lig index in the .mol2
    fmol2 = open(mol2Name, "r")
    lines = fmol2.readlines()
    indexForLigVal = 0
    isAtomSec = False
    for i, aline in enumerate(lines):
        if aline.startswith("@<TRIPOS>ATOM"):
            isAtomSec = True
        if aline.startswith("@<TRIPOS>BOND"):
            isAtomSec = False
        if not isAtomSec:
            continue
        splitted = aline.strip().split()
        if len(splitted) != 9:
            continue
        xpos = float(splitted[2])
        ypos = float(splitted[3])
        zpos = float(splitted[4])
        atomtype = splitted[5].split(".")[0]
        if atomtype in twoDic:
            atomtype = twoDic[atomtype]
        # if atomtype == "H":
        #     continue

        dicPOS_mol2[indexForLigVal] = (atomtype, xpos, ypos, zpos)
        listOfAtypes.append(atomtype)
        ligPOS_Index["_".join([atomtype, str(xpos), str(ypos), str(zpos)])] = indexForLigVal
        indexForLigVal += 1
    fmol2.close()
    return dicPOS_mol2, listOfAtypes, ligPOS_Index


# pocket pdb
# returns position dict and list of atoms
# _pocket.pdb and the original pdb file have not many things in common ==> position is common ==> key:position
def readPocketpdb(pdbName):
    dicPOS_pro = {}
    listOfAtypes = []
    pocPOS_Index = {}  # key:positions in str, value:pocket index in the pocket.pdb
    indexForPocVal = 0
    f = open(pdbName, "r")
    lines = f.readlines()
    for i, aline in enumerate(lines):
        if not aline.startswith("ATOM") and not aline.startswith("HETATM"):
            continue

        splitted = aline.strip().split()
        # print(pdbName, splitted)

        atom = splitted[-1] + splitted[1]  # to distinguish atoms with atom order
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

        atomtype = splitted[-1]
        if atomtype == "H":
            continue
        dicPOS_pro[indexForPocVal] = (atomtype, xpos, ypos, zpos)
        listOfAtypes.append(atomtype)
        pocPOS_Index["_".join([atomtype, str(xpos), str(ypos), str(zpos)])] = indexForPocVal
        indexForPocVal += 1
    f.close()
    return dicPOS_pro, listOfAtypes, pocPOS_Index


def euclidDist(ligPos, pocPos):
    # ligtypeList = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
    # poctypeList = ["C", "N", "O", "S"]

    ligtype, ligx, ligy, ligz = ligPos
    poctype, pocx, pocy, pocz = pocPos
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
def char2indices(listStr, dicC2I, seq_size):
    listIndices = [0] * seq_size
    charlist = listStr
    size = len(listStr)
    twoChars = {"Al": 1, "Au": 1, "Ag": 1, "As": 1, "Ba": 1, "Be": 1, "Bi": 1, "Br": 1, "Ca": 1, "Cd": 1, "Cl": 1,
                "Co": 1, "Cr": 1, "Cu": 1, "Dy": 1, "Eu": 1, "Fe": 1, "Ga": 1, "Gd": 1, "Ge": 1, "Hg": 1, "In": 1,
                "Ir": 1, "Li": 1, "Mg": 1, "Mn": 1, "Mo": 1, "Na": 1, "Ni": 1, "Nd": 1, "Pb": 1, "Pt": 1, "Pd": 1,
                "Rb": 1, "Ru": 1, "Sb": 1, "Se": 1, "Si": 1, "Sn": 1, "Sr": 1, "Ti": 1, "Tl": 1, "Yb": 1, "Zn": 1,
                "Zr": 1}
    prevTwoCharsFlag = False
    indexForList = 0
    for i, c in enumerate(charlist):
        if prevTwoCharsFlag:
            prevTwoCharsFlag = False
            continue

        if i != size - 1 and "".join(charlist[i:i + 2]) in twoChars:
            two = "".join(charlist[i:i + 2])
            if two not in dicC2I:
                dicC2I[two] = len(dicC2I) + 1
                listIndices[indexForList] = dicC2I[two]
                indexForList += 1
            else:
                listIndices[indexForList] = dicC2I[two]
                indexForList += 1
            prevTwoCharsFlag = True
        else:
            if c not in dicC2I:
                dicC2I[c] = len(dicC2I) + 1
                listIndices[indexForList] = dicC2I[c]
                indexForList += 1
            else:
                listIndices[indexForList] = dicC2I[c]
                indexForList += 1
    return listIndices
"""


# from pdb protein structure, get ligand index list for bond extraction and atom's index in the pocket.pdb
def get_atoms_from_pdb(ligPOS_Index, pdbPath, pocPOS_Index):
    # ligand side
    ori_ligand_dic = {}  # key=atom's serial number, value=lig atom index in the sdf
    # protein(pocket) side
    ori_pocket_dic = {}  # key=atom's serial number, value=pocket atom index in the pocket.pdb
    f = open(pdbPath, "r")
    lines = f.readlines()
    for i, aline in enumerate(lines):
        if not aline.startswith("ATOM") and not aline.startswith("HETATM"):
            continue

        splitted = aline.strip().split()
        atomSerialNum = splitted[1]
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

        atomtype = splitted[-1]
        tmpkey = "_".join([atomtype, str(xpos), str(ypos), str(zpos)])
        if tmpkey in ligPOS_Index:
            ori_ligand_dic[atomSerialNum] = ligPOS_Index[tmpkey]

        if tmpkey in pocPOS_Index:
            ori_pocket_dic[atomSerialNum] = pocPOS_Index[tmpkey]
    f.close()

    if len(ori_ligand_dic) != 0:
        return ori_ligand_dic, ori_pocket_dic  # , atom_name_list
    else:
        return None, None


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

        # if measure == "IC50":
        #     continue

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
        if value <= 4 or value >= 12:
            print('value is outside of typical range')
            continue
        if not os.path.exists("../pdbbind_files/" + pdbid):  # some pdbid only exists in the index files
            continue
        if not os.path.exists("../pdb_files/" + pdbid + ".pdb"):  # some pdbid only exists in the index files
            continue

        # read data from pdbbind files
        # these lines are required since pdbbind has only 1 ligand and 1 pocket data even if there are 3 interactions
        # get key:positions in str, value:lig index dict from the .sdf
        ligand_dict, listOfLatypes, ligPOS_Index = readmol2(
            "../pdbbind_files/" + pdbid + "/" + pdbid + "_ligand.mol2")
        # get key:positions in str, value:pocket index dict from the pocket.pdb
        pocket_dict, listOfPatypes, pocPOS_Index = readPocketpdb(
            "../pdbbind_files/" + pdbid + "/" + pdbid + "_pocket2.pdb")
        if len(listOfPatypes) >= max_num_seq or len(listOfLatypes) >= max_num_atoms:
            print('PocketPatoms length is over ' + str(max_num_seq) + ' or len(ligand) is over ' + str(
                max_num_atoms) + ' ==> we disregard such instances')
            continue

        vecOfLatoms = np.zeros([max_num_atoms])
        for i, Ltype in enumerate(listOfLatypes):
            vecOfLatoms[i] = dicAtom2I[atom_features(Ltype)]
        vecOfPatoms = np.zeros([max_num_seq])
        for i, Ptype in enumerate(listOfPatypes):
            vecOfPatoms[i] = dicAtom2I[atom_features(Ptype)]

        # read mol from sdf and make graph structure
        sdfMOLs = Chem.rdmolfiles.MolFromMol2File("../pdbbind_files/" + pdbid + "/" + pdbid + "_ligand.mol2",
                                                  sanitize=False, removeHs=False)
        if not sdfMOLs:
            sdfMOLs = Chem.SDMolSupplier("../pdbbind_files/" + pdbid + "/" + pdbid + "_ligand.sdf")[0]
            if not sdfMOLs:
                print("not a valid mol-ligand1")
                continue
        pocket_mol = Chem.rdmolfiles.MolFromPDBFile("../pdbbind_files/" + pdbid + "/" + pdbid + "_pocket2.pdb",
                                                    sanitize=False, removeHs=False, proximityBonding=False)
        if not pocket_mol:
            print("not a valid pocket_mol1")
            continue
        pocket_mol = Chem.rdmolops.RemoveHs(pocket_mol, sanitize=False)

        # vec_all_atoms, adj_mat1, n_l_atoms = Mol2Graph(sdfMOLs, pocket_mol)
        # if len(vec_all_atoms) == 0:
        #     print("not a valid mol2")
        #     continue

        # extract interaction info
        ori_ligand_dic, ori_pocket_dic = get_atoms_from_pdb(ligPOS_Index, "../pdb_files/" + pdbid + ".pdb",
                                                            pocPOS_Index)  # for bond atom identification
        if ori_ligand_dic is None:
            print('no such ligand in pdb', 'pdbid', pdbid, 'ligand', ligand)
            continue
        # assert len(pocPOS_Index) == len(ori_pocket_dic)<== this is different because _pocket.pdb cotains hydrogen atoms (random, maybe?)

        pairwise_mat = np.zeros([max_num_atoms, max_num_seq], dtype=np.int32)
        pairwise_exist = 0
        if os.path.exists('../plip_results/' + pdbid + 'out.txt'):  # plip have failed to produce results from some pdb
            pairwise_exist = get_interacts(pdbid, ligand, ori_ligand_dic, ori_pocket_dic, pairwise_mat)

        # either KIKD or IC50
        if measure in ['Ki', 'Kd']:
            if pdbid in casfList:
                datapack_test.append(
                    [vecOfLatoms, vecOfPatoms, value, pairwise_mat, pairwise_exist])
            else:
                datapack_kikd.append(
                    [vecOfLatoms, vecOfPatoms, value, pairwise_mat, pairwise_exist])
        elif measure == "IC50":
            datapack_ic50.append(
                [vecOfLatoms, vecOfPatoms, value, pairwise_mat, pairwise_exist])

    f.close()
    return datapack_kikd, datapack_ic50, datapack_test


# Model parameter intializer
def weights_init(m):
    for param in m.parameters():
        if param.dim() == 1:
            continue
        else:
            nn.init.xavier_normal_(param)
