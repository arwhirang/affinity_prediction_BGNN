import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, precision_score, roc_auc_score
import time
import torch
from torch import nn
import torch.nn.functional as F
from rdkit import Chem
import math
from metrics import *
import os

# atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
# bond_fdim = 6
max_num_bonds = 100
max_nb = 6
max_num_seq = 1000
max_num_atoms = 150


# embedding selection function requires 1-D indicies
def add_index(input_array, ebd_size):
    batch_size, n_vertex, n_nbs = np.shape(input_array)
    add_idx = np.array(list(range(0, (ebd_size) * batch_size, ebd_size)) * (n_nbs * n_vertex))
    add_idx = np.transpose(add_idx.reshape(-1, batch_size))
    add_idx = add_idx.reshape(-1)
    new_array = input_array.reshape(-1) + add_idx
    return new_array


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        return len(allowable_set) - 1
    return allowable_set.index(x)


def atom_features(atom):
    elem_list = ['<PAD>', 'B', 'C', 'F', 'H', 'I', 'K', 'N', 'O', 'P', 'S', 'U', 'V', 'W', 'Ag', 'Al', 'As', 'Au', 'Ba',
                 'Be', 'Bi', 'Br', 'Ca', 'Cd', 'Ce', 'Cl', 'Co', 'Cr', 'Cs', 'Cu', 'Dy', 'Eu', 'Fe', 'Ga', 'Gd', 'Ge',
                 'Hf', 'Hg', 'In', 'Ir', 'Li', 'Mg', 'Mn', 'Mo', 'Na', 'Nb', 'Ni', "Nd", "Pb", "Pd", "Pt", "Rb", 'Os',
                 'Re', 'Rh', "Ru", "Sb", "Se", "Si", 'Sm', "Sn", "Sr", 'Tc', 'Te', "Ti", "Tl", "Yb", "Zn", "Zr",
                 'unknown']
    return onek_encoding_unk(atom, elem_list)
    # + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5])
    # + onek_encoding_unk(atom.GetExplicitValence(), [1,2,3,4,5,6])
    # + onek_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5])
    # + [atom.GetIsAromatic()], dtype=np.float32)


def bond_features(bond):
    bt = bond.GetBondType()
    return np.array(
        [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, \
         bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()], dtype=np.float32)


def Mol2Graph(mol, dicAtom2I, dicBond2I, listOfLatypes):
    # convert molecule to GNN input
    n_atoms = mol.GetNumAtoms()
    assert mol.GetNumBonds() >= 0
    n_bonds = max(mol.GetNumBonds(), 1)
    if n_atoms > max_num_atoms or n_bonds > max_num_bonds:
        return [], [], [], [], []
    vecOfLatoms = np.zeros((max_num_atoms,), dtype=np.int32)  # atom feature ID
    fbonds = np.zeros((max_num_bonds,), dtype=np.int32)  # bond feature ID
    atom_nb = np.zeros((max_num_atoms, max_nb), dtype=np.int32)
    bond_nb = np.zeros((max_num_atoms, max_nb), dtype=np.int32)
    num_nbs = np.zeros((max_num_atoms,), dtype=np.int32)
    num_nbs_mat = np.zeros((max_num_atoms, max_nb), dtype=np.int32)

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if listOfLatypes[idx] != atom.GetSymbol():
            print(listOfLatypes)
            for atom in mol.GetAtoms():
                print(atom.GetSymbol(), end=", ")
        if listOfLatypes[idx] != atom.GetSymbol():
            return [], [], [], [], []
        assert listOfLatypes[idx] == atom.GetSymbol()
        vecOfLatoms[idx] = dicAtom2I[atom_features(atom.GetSymbol())]
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        idx = bond.GetIdx()
        fbonds[idx] = dicBond2I[''.join(str(x) for x in bond_features(bond).astype(int).tolist())]
        try:
            atom_nb[a1, num_nbs[a1]] = a2
            atom_nb[a2, num_nbs[a2]] = a1
        except:
            return [], [], [], [], []
        bond_nb[a1, num_nbs[a1]] = idx
        bond_nb[a2, num_nbs[a2]] = idx
        num_nbs[a1] += 1
        num_nbs[a2] += 1
    for i in range(len(num_nbs)):
        num_nbs_mat[i, :num_nbs[i]] = 1
    return vecOfLatoms, fbonds, atom_nb, bond_nb, num_nbs_mat


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
                    continue  # this bond contains other ligand-pocket chain
                # bond_list.append((bond_type+'_'+str(len(bond_list)), aa_chain, aa_name, aa_id, [atom_idx_protein], ori_pocket_dic[atom_idx_protein], ligand_chain, ligand_name, ligand_id, [atom_idx_ligand]))
                # interact_ligand_indicies.append(ori_ligand_dic[atom_idx_ligand])
                # interact_pocket_indicies.append(ori_pocket_dic[atom_idx_protein])
                pairwise_mat[ori_ligand_dic[atom_idx_ligand]][ori_pocket_dic[atom_idx_protein]] = 1
                pairwise_exist = 1
            elif bond_type == 'Hydrophobic Interactions':
                continue
                # atom_idx_ligand, atom_idx_protein = lines[8], lines[9]
                # if atom_idx_ligand not in ori_ligand_dic:
                #     continue#this bond contains other ligand-pocket chain
                # bond_list.append((bond_type+'_'+str(len(bond_list)), aa_chain, aa_name, aa_id, [atom_idx_protein], ori_pocket_dic[atom_idx_protein], ligand_chain, ligand_name, ligand_id, [atom_idx_ligand]))
                # interact_ligand_indicies.append(ori_ligand_dic[atom_idx_ligand])
                # interact_pocket_indicies.append(ori_pocket_dic[atom_idx_protein])
                # pairwise_mat[ori_ligand_dic[atom_idx_ligand]][ori_pocket_dic[atom_idx_protein]] = 1
                # pairwise_exist = 1
            elif bond_type in ['pi-Stacking', 'pi-Cation Interactions']:
                continue  # no protein atom involved
            #                 atom_idx_ligand_list = list(map(int, lines[11].split(',')))
            #                 if len(set(atom_idx_ligand_list).intersection(set(atom_idx_list))) != len(atom_idx_ligand_list):
            #                     print(bond_type, 'error: atom index in plip result not in atom_idx_list')
            #                     print(atom_idx_ligand_list)
            #                     return None
            #                 bond_list.append((bond_type+'_'+str(len(bond_list)), aa_chain, aa_name, aa_id, [], ligand_chain, ligand_name, ligand_id, atom_idx_ligand_list))
            elif bond_type == 'Salt Bridges':
                continue  # no protein atom involved
            #                 atom_idx_ligand_list = list(set(map(int, lines[10].split(','))))
            #                 if len(set(atom_idx_ligand_list).intersection(set(atom_idx_list))) != len(atom_idx_ligand_list):
            #                     print('error: atom index in plip result not in atom_idx_list')
            #                     print('Salt Bridges', atom_idx_ligand_list, set(atom_idx_ligand_list).intersection(set(atom_idx_list)))
            #                     return None
            #                 bond_list.append((bond_type+'_'+str(len(bond_list)), aa_chain, aa_name, aa_id, [], ligand_chain, ligand_name, ligand_id, atom_idx_ligand_list))
            elif bond_type == 'Halogen Bonds':
                continue
                # atom_idx1, atom_idx2 = lines[11], lines[13]
                # if atom_idx1 in ori_ligand_dic and atom_idx2 in ori_ligand_dic:   # discard ligand-ligand interaction
                #     continue
                # if atom_idx1 in ori_ligand_dic:
                #     atom_idx_ligand, atom_idx_protein = atom_idx1, atom_idx2
                # elif atom_idx2 in ori_ligand_dic:
                #     atom_idx_ligand, atom_idx_protein = atom_idx2, atom_idx1
                # else:
                #     continue  # this bond contains other ligand-pocket chain
                # bond_list.append((bond_type+'_'+str(len(bond_list)), aa_chain, aa_name, aa_id, [atom_idx_protein], ori_pocket_dic[atom_idx_protein], ligand_chain, ligand_name, ligand_id, [atom_idx_ligand]))
                # interact_ligand_indicies.append(ori_ligand_dic[atom_idx_ligand])
                # interact_pocket_indicies.append(ori_pocket_dic[atom_idx_protein])
                # pairwise_mat[ori_ligand_dic[atom_idx_ligand]][ori_pocket_dic[atom_idx_protein]] = 1
                # pairwise_exist = 1
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
                # if isAlreadyZero:
                #     print("ok, this is wrong!")
            # else:
            #     isAlreadyZero = True
        # a[i,:len(arr)] = 1#arr.shape[0]
    return a


# returns position dict and list of atoms
def readsdf(sdfName):
    dicPOS_sdf = {}
    listOfAtypes = []
    listOfAtoms = []
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
        if atomtype == "H":
            continue
        atom = splitted[3] + str(i)  # to distinguish atoms with atom order?
        xpos = float(splitted[0])
        ypos = float(splitted[1])
        zpos = float(splitted[2])

        dicPOS_sdf[atom] = (atomtype, xpos, ypos, zpos)
        listOfAtypes.append(atomtype)
        listOfAtoms.append(atom)
        ligPOS_Index["_".join([atomtype, str(xpos), str(ypos), str(zpos)])] = indexForLigVal
        indexForLigVal += 1
    fsdf.close()
    return dicPOS_sdf, listOfAtypes, listOfAtoms, ligPOS_Index


# pocket pdb
# returns position dict and list of atoms
# _pocket.pdb and the original pdb file have not many things in common ==> position is common ==> key:position
def readPocketpdb(pdbName):
    dicPOS_pro = {}
    listOfAtypes = []
    listOfAtoms = []
    pocPOS_Index = {}  # key:positions in str, value:pocket index in the pocket.pdb
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
        dicPOS_pro[atom] = (atomtype, xpos, ypos, zpos)
        listOfAtypes.append(atomtype)
        listOfAtoms.append(atom)
        pocPOS_Index["_".join([atomtype, str(xpos), str(ypos), str(zpos)])] = i
    f.close()
    return dicPOS_pro, listOfAtypes, listOfAtoms, pocPOS_Index


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
    twoDic = {"AL": "Al", "AU": "Au", "AG": "Ag", "AS": "As", "BA": "Ba", "BE": "Be", "BI": "Bi", "BR": "Br",
              "CA": "Ca", "CD": "Cd", "CL": "Cl", "CO": "Co", "CR": "Cr", "CU": "Cu", "DY": "Dy", "EU": "Eu",
              "FE": "Fe", "GA": "Ga", "GD": "Gd", "GE": "Ge", "HG": "Hg", "IN": "In", "IR": 'Ir', "LI": "Li",
              "MG": "Mg", "MN": "Mn", "MO": "Mo", "NA": "Na", "ND": "Nd", "NI": "Ni", "PB": "Pb", "PD": "Pd",
              "PT": "Pt", "RB": "Rb", "RU": "Ru", "SB": "Sb", "SE": "Se", "SI": "Si", "SN": "Sn", "SR": "Sr",
              "TI": "Ti", "TL": "Tl", "YB": "Yb", "ZN": "Zn", "ZR": "Zr"}

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
    evee = 5
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


def load_data2(MEASURE, dicAtom2I, dicBond2I):
    datapack = []
    f = open('../data/pdbbind_all_datafile.tsv')
    for line in f.readlines():
        pdbid, pid, cid, inchi, seq, measure, label = line.strip().split('\t')
        # filter interaction type and invalid molecules
        # either KIKD or IC50
        if MEASURE == 'KIKD':
            if measure not in ['Ki', 'Kd']:
                continue
        elif measure != MEASURE:
            continue

        # get labels
        value = float(label)
        if value <= 4 or value >= 11:
            print('value is outside of typical range')
            continue
        if not os.path.exists("../pdbbind_files/" + pdbid):  # some pdbid only exists in the index files
            continue

        # read data from pdbbind files
        # these lines are required since pdbbind has only 1 ligand and 1 pocket data even if there are 3 interactions
        # get key:positions in str, value:lig index dict from the .sdf
        ligand_dict, listOfLatypes, listOfLatoms, ligPOS_Index = readsdf(
            "../pdbbind_files/" + pdbid + "/" + pdbid + "_ligand.sdf")
        # get key:positions in str, value:pocket index dict from the pocket.pdb
        pocket_dict, listOfPatypes, listOfPatoms, pocPOS_Index = readPocketpdb(
            "../pdbbind_files/" + pdbid + "/" + pdbid + "_pocket2.pdb")
        if len(listOfPatypes) >= max_num_seq or len(listOfLatoms) >= max_num_atoms:
            print('PocketPatoms length is over ' + str(max_num_seq) + ' or len(ligand) is over ' + str(
                max_num_atoms) + ' ==> we disregard such instances')
            continue

        # read mol from sdf and make graph structure
        sdfMOLs = Chem.rdmolfiles.MolFromMol2File("../pdbbind_files/" + pdbid + "/" + pdbid + "_ligand.mol2")
        if not sdfMOLs:
            print("not a valid mol")
            continue
        vecOfLatoms, fb, anb, bnb, nbs_mat = Mol2Graph(sdfMOLs, dicAtom2I, dicBond2I, listOfLatypes)
        if len(vecOfLatoms) == 0:
            print("not a valid mol")
            continue
        # extract interaction info
        ori_ligand_dic, ori_pocket_dic = get_atoms_from_pdb(ligPOS_Index, "../pdb_files/" + pdbid + ".pdb",
                                                            pocPOS_Index)  # for bond atom identification
        if ori_ligand_dic is None:
            print('no such ligand in pdb', 'pdbid', pdbid, 'cid', cid)
            continue
        # assert len(pocPOS_Index) == len(ori_pocket_dic)<== this is different because _pocket.pdb cotains hydrogen atoms (random, maybe?)

        pairwise_mat = np.zeros([max_num_atoms, max_num_seq], dtype=np.int32)
        pairwise_exist = 0
        if os.path.exists('../plip_results/' + pdbid + 'out.txt'):  # plip have failed to produce results from some pdb
            pairwise_exist = get_interacts(pdbid, cid, ori_ligand_dic, ori_pocket_dic, pairwise_mat)

        # calculate fri matrix
        tmpFRIMat = np.zeros([max_num_atoms, max_num_seq])
        cutoffDist = 20  # angstroms
        for i, ligKey in enumerate(listOfLatoms):
            for j, pocKey in enumerate(listOfPatoms):
                eucDist = euclidDist(ligand_dict[ligKey], pocket_dict[pocKey])
                if eucDist > cutoffDist:
                    currVal = 0
                else:
                    currVal = fri(listOfLatypes[i], listOfPatypes[j], eucDist)
                tmpFRIMat[i][j] = currVal
        datapack.append(
            [vecOfLatoms, fb, anb, bnb, nbs_mat, listOfPatypes, pdbid, value, tmpFRIMat, pairwise_mat, pairwise_exist])
    f.close()
    return datapack


# Model parameter intializer
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        # nn.init.normal_(m.weight.data, mean=0, std=min(1.0 / math.sqrt(m.weight.data.shape[-1]), 0.1))
        # nn.init.constant_(m.bias, 0)


# Custom loss
class Masked_BCELoss(nn.Module):
    def __init__(self):
        super(Masked_BCELoss, self).__init__()
        self.criterion = nn.BCELoss(reduce=None)

    def forward(self, pred, label, vertex_mask, seq_mask, pairwise_exist):
        batch_size = pred.size(0)
        loss_all = self.criterion(pred, label)
        loss_mask = torch.matmul(vertex_mask.view(batch_size, -1, 1),
                                 seq_mask.view(batch_size, 1, -1))  # * pairwise_exist.view(-1, 1, 1)
        loss = torch.sum(loss_all * loss_mask)  # / torch.sum(pairwise_exist).clamp(min=1e-10)
        return loss





