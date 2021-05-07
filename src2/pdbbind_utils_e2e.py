import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
from scipy import stats
import torch
from torch import nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
import math
import os
import pandas as pd
from scipy.spatial.distance import cdist

MAX_NUM_SEQ = 2000
MAX_NUM_ATOMS = 200
MAX_CONFS = 30


# ----------------- modified code from ECIF start -----------------


def GetAtomType(atom):
    # This function takes an atom in a molecule and returns its type as defined for ECIF
    AtomType = [atom.GetSymbol(),
                str(atom.GetExplicitValence()),
                str(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() != "H"])),
                str(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() == "H"])),
                str(int(atom.GetIsAromatic())),
                str(int(atom.IsInRing())),
                ]
    return ";".join(AtomType)


def LoadSDFasDF(SDF):
    lig_atom_list = ['C', 'N', 'O', 'S', 'F', 'P', 'Ca', 'Cl', 'Cu', 'Br', 'B', 'I']
    # This function takes an SDF for a ligand as input and returns it as a pandas DataFrame with its atom types
    # labeled according to ECIF
    m = Chem.MolFromMolFile(SDF, sanitize=False)
    m.UpdatePropertyCache(strict=False)

    ECIF_atoms = []
    for atom in m.GetAtoms():
        if atom.GetSymbol() in lig_atom_list:  # Include only certain heavy atoms
            entry = [int(atom.GetIdx()), GetAtomType(atom)]
            pos = m.GetConformer().GetAtomPosition(atom.GetIdx())
            entry.append(float("{0:.4f}".format(pos.x)))
            entry.append(float("{0:.4f}".format(pos.y)))
            entry.append(float("{0:.4f}".format(pos.z)))
            ECIF_atoms.append(entry)

    df = pd.DataFrame(ECIF_atoms)
    df.columns = ["ATOM_INDEX", "ECIF_ATOM_TYPE", "X", "Y", "Z"]
    return df


def LoadPDBasDF(PDB, Atom_Keys):
    # This function takes a PDB for a protein as input and returns it as a pandas DataFrame with its atom types
    # labeled according to ECIF
    ECIF_atoms = []
    f = open(PDB)
    for i in f:
        if i[:4] == "ATOM":
            # Include only non-hydrogen atoms
            if (len(i[12:16].replace(" ", "")) < 4 and i[12:16].replace(" ", "")[0] != "H") or (
                    len(i[12:16].replace(" ", "")) == 4 and i[12:16].replace(" ", "")[1] != "H" and
                    i[12:16].replace(" ", "")[0] != "H"):
                ECIF_atoms.append([int(i[6:11]), i[17:20] + "-" + i[12:16].replace(" ", ""), float(i[30:38]),
                                   float(i[38:46]), float(i[46:54])])
    f.close()

    df = pd.DataFrame(ECIF_atoms, columns=["ATOM_INDEX", "PDB_ATOM", "X", "Y", "Z"])
    df = df.merge(Atom_Keys, left_on='PDB_ATOM', right_on='PDB_ATOM')[
        ["ATOM_INDEX", "ECIF_ATOM_TYPE", "X", "Y", "Z"]].sort_values(by="ATOM_INDEX").reset_index(drop=True)
    return df


def GetPLPairs(PDB_protein, SDF_ligand, Atom_Keys, distance_cutoff=6.5):
    # This function returns the protein-ligand atom-type pairs for a given distance cutoff
    # Load both structures as pandas DataFrames
    Target = LoadPDBasDF(PDB_protein, Atom_Keys)
    Ligand = LoadSDFasDF(SDF_ligand)

    # Take all atoms from the target within a cubic box around the ligand considering the "distance_cutoff criterion"
    for i in ["X", "Y", "Z"]:
        Target = Target[Target[i] < float(Ligand[i].max()) + distance_cutoff]
        Target = Target[Target[i] > float(Ligand[i].min()) - distance_cutoff]

    if len(Target) >= MAX_NUM_SEQ or len(Ligand) >= MAX_NUM_ATOMS:
        return False, [], [], []

    Distances = cdist(Ligand[["X", "Y", "Z"]], Target[["X", "Y", "Z"]], metric="euclidean")

    return True, list(Ligand["ECIF_ATOM_TYPE"]), list(Target["ECIF_ATOM_TYPE"]), Distances


# ----------------- modified code from ECIF end -----------------

# ----------------- loading re-docking data codes start -----------------

# name says it all
def defaultdic_action(k, _dic, isTest=False):
    if k in _dic:
        return _dic[k]
    else:
        if isTest:
            return 1
        else:
            _dic[k] = len(_dic) + 1
            return _dic[k]


# distance is used for weight matrix. binning is effective ML method for dealing with continuous value
def cutoffBinning(eucdist, thres):  # thres starts from 6.95 and ends at 2.75
    for i, ele in enumerate(thres):
        if eucdist > ele:
            return i / 30.0
    return 1


# distance for binning
def makeThreshold():
    thres = list(np.linspace(2.75, 6, 30))  # usually 0.06 range for each bin
    for i, ele in enumerate(thres):
        thres[i] = math.floor(ele * 100) / 100
    thres.reverse()  # starts from 6.95 and ends at 2.75
    return thres


# load ligand, protein, and adjacency matrix while pre-processing ECIF
def plp_save(lig_confs, pro_confs, adj_confs, pdbid, subdir, subfname, Atom_Keys, dic_atom2i):
    is_valid, ligand, target, distances = GetPLPairs(
        "../../pdbbind_files/" + pdbid + "/" + pdbid + "_protein.pdb",
        subdir + "/" + subfname,
        # "../../pdbbind_files/" + pdbid + "/" + pdbid + "_ligand.sdf",
        Atom_Keys, distance_cutoff=6.0)

    if not is_valid:
        print("too long sdf or too long pdb")
        return is_valid

    # convert the ECIF features to an index of a dictionary - ligand
    vec_lig1 = np.zeros([len(ligand)])
    for i, ele in enumerate(ligand):
        vec_lig1[i] = defaultdic_action(ele, dic_atom2i)
    # convert the ECIF features to an index of a dictionary - protein pocket
    vec_poc1 = np.zeros([len(target)])
    for i, ele in enumerate(target):
        vec_poc1[i] = defaultdic_action(ele, dic_atom2i)

    # apply binning to the adj distance matrix
    adj_inter = distances
    # print(adj_inter.shape)
    thres = makeThreshold()
    cutoffDist = 6  # angstroms
    for i in range(len(ligand)):
        for j in range(len(target)):
            eucDist = adj_inter[i][j]
            if eucDist > cutoffDist:
                currVal = 0
            else:
                currVal = cutoffBinning(eucDist, thres)
            adj_inter[i][j] = currVal

    lig_confs.append(vec_lig1)
    pro_confs.append(vec_poc1)
    adj_confs.append(adj_inter)
    return is_valid


# main code for loading data
def load_data(dic_atom2i):
    #  datapack_test = []
    #  datapack_kikd = []
    datapack_ic50 = []

    Atom_Keys = pd.read_csv("PDB_Atom_Keys.csv", sep=",")  # PDB_atom_keys is a file from ECIF

    f = open('../pdbbind_index/INDEX_all.2019')
    for line in f.readlines():
        if line[0] == '#':
            continue

        # filter erroneous or weirdly complex data
        ligand = line.strip().split('(')[1].split(')')[0]
        if '-mer' in ligand:
            continue
        elif len(ligand) != 3:
            continue

        lines = line.split('/')[0].strip().split('  ')
        pdbid = lines[0]
        if '~' in lines[3]:
            continue
        elif '<' in lines[3]:
            continue
        elif '>' in lines[3]:
            continue
        else:
            measure = lines[3].split('=')[0]
            value = float(lines[3].split('=')[1][:-2])
            unit = lines[3].split('=')[1][-2:]

        if not os.path.exists("../pdbbind_files/" + pdbid):  # some pdbid only exists in the index files
            continue

        if not os.path.exists("../pdbbind_files/" + pdbid + "/sep_sdfs"):  # some pdbid does not have predicted confs
            continue

        if measure != "IC50":
            continue

        if unit == 'nM':
            pvalue = unit_to_kcal(value, 1e-9)  # uM is 1e-6
        elif unit == 'uM':
            pvalue = unit_to_kcal(value, 1e-6)  # uM is 1e-6
        elif unit == 'mM':
            pvalue = unit_to_kcal(value, 1e-3)  # uM is 1e-6
        elif unit == 'pM':
            pvalue = unit_to_kcal(value, 1e-12)  # uM is 1e-6
        elif unit == 'fM':
            pvalue = unit_to_kcal(value, 1e-15)  # uM is 1e-6
        
        # IC50 cases
        if pvalue < -13 or pvalue > -5:  # discard weird val
            continue

        pvalue = pvalue + 9  # -9.x is pdbbind mean

        # get labels
        value = float(pvalue)

        # stored whole conformer of a pdbid's ligand, protein, adjacency
        lig_confs = []  # np.zeros((MAX_CONFS, MAX_NUM_ATOMS))
        pro_confs = []  # np.zeros((MAX_CONFS, MAX_NUM_SEQ))
        adj_confs = []  # np.zeros((MAX_CONFS, MAX_NUM_ATOMS, MAX_NUM_SEQ))

        # load original data
        ori_lig_dir = "../pdbbind_files/" + pdbid
        ori_lig_fname = pdbid + "_ligand.sdf"
        is_valid = plp_save(lig_confs, pro_confs, adj_confs, pdbid, ori_lig_dir, ori_lig_fname, Atom_Keys,
                            dic_atom2i)

        # load re-docking data
        subdir = "../pdbbind_files/" + pdbid + "/sep_sdfs"
        subfiles = [name for name in os.listdir(subdir)]
        for i, subfname in enumerate(subfiles):
            if i == MAX_CONFS:
                break
            if not is_valid:
                break
            is_valid = plp_save(lig_confs, pro_confs, adj_confs, pdbid, subdir, subfname, Atom_Keys,
                                dic_atom2i)

        if not is_valid:
            continue
        if not lig_confs:  # catch weird cases that no sub conformers are found
            continue

        # either KIKD or IC50. Do not use both at once!
        if measure in ['Ki', 'Kd']:
            if pdbid in casfList:
                datapack_test.append([lig_confs, pro_confs, adj_confs, value, 0])
            else:
                datapack_kikd.append([lig_confs, pro_confs, adj_confs, value, 0])
        elif measure == "IC50":
            # if pdbid in pre_validset:
            #     datapack_test.append([lig_confs, pro_confs, des_confs, adj_confs, value, 0])
            # else:
            datapack_ic50.append([lig_confs, pro_confs, adj_confs, value, 0])

    f.close()
    return datapack_kikd, datapack_ic50, datapack_test

# ----------------- loading re-docking data codes end -----------------

# ----------------- loading cross-docking data codes start -----------------


def subproc(pdb_file, ligdir, cur_dir, tmpdatapack, Atom_Keys, dic_atom2i, gc4_id, gc4_label, id_idx):
    fnames = []
    for (dirpath, dirnames, _filename) in os.walk(ligdir + cur_dir):
        fnames.extend(_filename)
        break

    cur_idx = -1
    for i, ele in enumerate(gc4_id):
        if ele == id_idx:
            cur_idx = i
            break
    if cur_idx == -1:
        print("error! no matching idx found!")

    # stored whole conformer of a pdbid's ligand, protein, adjacency
    lig_confs = []  # np.zeros((MAX_CONFS, MAX_NUM_ATOMS))
    pro_confs = []  # np.zeros((MAX_CONFS, MAX_NUM_SEQ))
    adj_confs = []  # np.zeros((MAX_CONFS, MAX_NUM_ATOMS, MAX_NUM_SEQ))
    for i, _fname in enumerate(fnames):  # sdf name
        if i == MAX_CONFS:
            break
        # custom GetPLPairs func
        is_valid, ligand, target, distances = GetPLPairs(pdb_file, ligdir + cur_dir + "/" + _fname, Atom_Keys,
                                                         distance_cutoff=6.0)
        if not is_valid:
            print("too long sdf or too long pdb, well...")
            continue

        # convert the ECIF features to index of a dictionary - ligand
        vec_lig1 = np.zeros([len(ligand)])
        for i, ele in enumerate(ligand):
            vec_lig1[i] = defaultdic_action(ele, dic_atom2i, isTest=True)
        # convert the ECIF features to index of a dictionary - protein
        vec_poc1 = np.zeros([len(target)])
        for i, ele in enumerate(target):
            vec_poc1[i] = defaultdic_action(ele, dic_atom2i, isTest=True)

        # apply binning to the adj distance matrix
        adj_inter = distances
        thres = makeThreshold()
        cutoffDist = 6  # angstroms
        for i in range(len(ligand)):
            for j in range(len(target)):
                eucDist = adj_inter[i][j]
                if eucDist > cutoffDist:
                    currVal = 0
                else:
                    currVal = cutoffBinning(eucDist, thres)
                adj_inter[i][j] = currVal

        lig_confs.append(vec_lig1)
        pro_confs.append(vec_poc1)
        adj_confs.append(adj_inter)

    if not lig_confs:
        print("no lig confs! <== usually no conformer was generated for: " + cur_dir + "/" + gc4_id[cur_idx])
        return

    tmpdatapack.append([lig_confs, pro_confs, adj_confs, gc4_label[cur_idx], gc4_id[cur_idx]])


def loadgc4BACEset(dic_atom2i):
    tmpdatapack = []
    # load scoreset label
    gc4_label_whole = pd.read_csv("BACE/BACE_score_compounds_D3R_GC4_answers.csv")  # for values
    gc4_label = list(gc4_label_whole[:]["Affinity"])
    gc4_id = list(gc4_label_whole[:]["Cmpd_ID"])
    for i, val in enumerate(gc4_label):
        # normalization
        pvalue = uM_to_kcal(val) + 9  # -9.x is pdbbind mean
        gc4_label[i] = pvalue

    # load matching ligand and reference data #
    ref_data = pd.read_csv('BACE/similar_pdbid_info2.tsv', header=None,
                           names=['d3r_id', 'd3r_smile', 'pdb_id', 'pdb_lig', 'pdb_smile', 'smarts'], sep='\t')

    for _, row in ref_data.iterrows():
        _id = row["d3r_id"]
        ref_pdbid = row["pdb_id"]
        refpro = 'BACE/' + str(ref_pdbid) + "_protein.pdb"
        idnum = str(_id[5:])

        # load data from predicted conformers
        ligdir = "BACE/"
        Atom_Keys = pd.read_csv("BACE/PDB_Atom_Keys.csv", sep=",")
        subproc(refpro, ligdir, idnum, tmpdatapack, Atom_Keys, dic_atom2i, gc4_id, gc4_label, _id)
    return tmpdatapack


def loadchemblBACEset(dic_atom2i):
    if os.path.exists("chembl_bace.pkl"):
        tmpdatapack = pickle.load(open('chembl_bace.pkl', 'rb'))
        return tmpdatapack

    tmpdatapack = []
    # load scoreset label
    label_whole = pd.read_csv("chembl_bace/BACE_IC50.csv")  # for values
    _label = list(label_whole[:]["Standard Value"])
    _id = list(label_whole[:]["Molecule ChEMBL ID"])
    for i, val in enumerate(_label):
        # normalization
        pvalue = nM_to_kcal(val) + 9  # -9.x is pdbbind mean
        _label[i] = pvalue

    # load matching ligand and reference data #
    ref_data = pd.read_csv('chembl_bace/similar_pdbid_info_bace.tsv', header=None,
                           names=['d3r_id', 'd3r_smile', 'pdb_id', 'pdb_lig', 'pdb_smile', 'smarts'], sep='\t')

    for _, row in ref_data.iterrows():
        d3r_id = row["d3r_id"]
        ref_pdbid = row["pdb_id"]
        refpro = 'chembl_bace/' + str(ref_pdbid) + "_protein.pdb"
        # load data from predicted conformers
        ligdir = "chembl_bace/"
        Atom_Keys = pd.read_csv("PDB_Atom_Keys.csv", sep=",")
        id_idx = d3r_id.split("_")[0]
        cur_dir = str(d3r_id.split("_")[1])
        subproc(refpro, ligdir, cur_dir, tmpdatapack, Atom_Keys, dic_atom2i, _id, _label, id_idx)
    pickle.dump(tmpdatapack, open('chembl_bace.pkl', 'wb'))
    return tmpdatapack


def loadgc3_CATSset(dic_atom2i):
    tmpdatapack = []

    # load scoreset label
    gc4_label_whole = pd.read_csv("gc3_CATS/final_CatS_score_compounds_D3R_GC3.csv")  # for values
    gc4_label = list(gc4_label_whole[:]["Affinity"])
    gc4_id = list(gc4_label_whole[:]["Cmpd_ID"])
    for i, val in enumerate(gc4_label):
        # normalization
        pvalue = uM_to_kcal(val) + 9  # -9.x is pdbbind mean
        gc4_label[i] = pvalue
        gc4_id[i] = str(gc4_id[i][5:])

    # load matching ligand and reference data #
    ref_data = pd.read_csv('gc3_CATS/similar_pdbid_info.tsv', header=None,
                           names=['d3r_id', 'd3r_smile', 'pdb_id', 'pdb_lig', 'pdb_smile', 'smarts'], sep='\t')

    for _, row in ref_data.iterrows():
        _id = row["d3r_id"]
        ref_pdbid = row["pdb_id"]
        refpro = 'gc3_CATS/' + str(ref_pdbid) + "_protein.pdb"
        idnum = str(_id[5:])

        # load data from predicted conformers
        ligdir = "gc3_CATS/"
        Atom_Keys = pd.read_csv("gc3_CATS/PDB_Atom_Keys.csv", sep=",")
        subproc(refpro, ligdir, idnum, tmpdatapack, Atom_Keys, dic_atom2i, gc4_id, gc4_label, idnum)
    return tmpdatapack


def loadwholeCATSset(dic_atom2i):
    tmpdatapack = []

    # load scoreset label
    gc4_label_whole = pd.read_csv("CATS/CatS_score_compounds_D3R_GC4_answers.csv")  # for values
    gc4_label = list(gc4_label_whole[:]["Affinity"])
    gc4_id = list(gc4_label_whole[:]["Cmpd_ID"])
    for i, val in enumerate(gc4_label):
        # normalization
        pvalue = uM_to_kcal(val) + 9  # -9.x is pdbbind mean
        gc4_label[i] = pvalue
        gc4_id[i] = str(gc4_id[i][5:])

    # load matching ligand and reference data #
    ref_data = pd.read_csv('CATS/similar_pdbid_info.tsv', header=None,
                           names=['d3r_id', 'd3r_smile', 'pdb_id', 'pdb_lig', 'pdb_smile', 'smarts'], sep='\t')

    for _, row in ref_data.iterrows():
        _id = row["d3r_id"]
        ref_pdbid = row["pdb_id"]
        refpro = 'CATS/' + str(ref_pdbid) + "_protein.pdb"
        idnum = str(_id[5:])

        # load data from predicted conformers
        ligdir = "CATS/"
        Atom_Keys = pd.read_csv("CATS/PDB_Atom_Keys.csv", sep=",")
        subproc(refpro, ligdir, idnum, tmpdatapack, Atom_Keys, dic_atom2i, gc4_id, gc4_label, idnum)
    return tmpdatapack


# ----------------- loading cross-docking data codes end -----------------

# -----------codes for utility start -----------

# rough evaluation
def reg_scores(label, pred):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    assert len(pred) == len(label)
    return mean_squared_error(label, pred, squared=False), stats.spearmanr(label, pred)[0]


def uM_to_kcal(ic50):
    # convert the ic50 values to kcal, original code from https://github.com/drugdata/D3R_grandchallenge_evaluator
    return math.log(ic50 * 1e-6) * 0.5961


def nM_to_kcal(ic50):
    # convert the ic50 values to kcal, original code from https://github.com/drugdata/D3R_grandchallenge_evaluator
    return math.log(ic50 * 1e-9) * 0.5961


def unit_to_kcal(ic50, unit):
    # convert the ic50 values to kcal, original code from https://github.com/drugdata/D3R_grandchallenge_evaluator
    return math.log(ic50 * unit) * 0.5961  # modified from 0.5961 to 1


# Model parameter intializer
def weights_init(m):
    for param in m.parameters():
        if param.dim() == 1:
            continue
        else:
            nn.init.xavier_normal_(param)

# -----------codes for utility end -----------
