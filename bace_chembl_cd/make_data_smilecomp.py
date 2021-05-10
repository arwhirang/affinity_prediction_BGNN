import pandas as pd
import os
from rdkit import Chem
import math
import numpy as np
from shutil import copyfile

pathToProjHome = "../"
TruthFilePath = "truth"


def extract_pdbid_from_name(target):
    retlist = []
    f = open(pathToProjHome + 'pdbbind_index/INDEX_all_name.2019')
    for line in f.readlines():
        splitted = line.strip().split('  ')
        pdbid = splitted[0]
        pro_id = splitted[2]
        if pro_id == target:
            retlist.append(pdbid)
    f.close()
    return retlist
    

def load_data():
    target_pdbs = extract_pdbid_from_name("P56817")  # P56817 :BACE
    print(target_pdbs)
    tmpdatapack = []
    
    arr1 = ["PDBID"]
    arr2 = ["SMILE"]
    arr3 = ["LIG_ID"]
    arr4 = ["affinity"]  # not used in smile_comp
    arr_col = arr1 + arr2 + arr3 + arr4

    checkSameSMILES = {}

    if not os.path.exists("truth"):
        os.mkdir("truth")

    f = open(pathToProjHome + 'pdbbind_index/INDEX_all.2019')
    for line in f.readlines():
        ligand_id = line.strip().split('(')[1].split(')')[0]
        lines = line.split('/')[0].strip().split('  ')
        pdbid = lines[0]
        
        if pdbid not in target_pdbs:
            continue
            
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
         
        # all thyroid receptor beta is ki/kd values   
        #if measure != "IC50":
        #    continue

        if not os.path.exists(pathToProjHome + "pdbbind_files/" + pdbid):  # some pdbid only exists in the index files
            print("pdbid not found")
            continue
            
        sdfMOLs = Chem.MolFromMolFile(pathToProjHome + "pdbbind_files/" + pdbid + "/" + pdbid + "_ligand.sdf", sanitize=False)
        if not sdfMOLs:
            print("not a valid mol-ligand1:", pdbid)
            continue

        pvalue = -np.log10(value) + 6
            
        # for truth ligand/protein file for conformer generation processes
        copyfile(pathToProjHome + "pdbbind_files/" + pdbid + "/" + pdbid + "_ligand.sdf", TruthFilePath + "/" + pdbid + "_ligand.sdf")
        copyfile(pathToProjHome + "pdbbind_files/" + pdbid + "/" + pdbid + "_protein.pdb", TruthFilePath + "/" + pdbid + "_protein.pdb")
            
        smi = Chem.MolToSmiles(sdfMOLs)
        print(smi)
        if smi not in checkSameSMILES:
            checkSameSMILES[smi] = 1
        else:
            continue

        tmpdatapack.append([pdbid] + [smi] + [ligand_id] + [pvalue])

        if len(tmpdatapack) == 10:
            break

    print(len(tmpdatapack))
    df = pd.DataFrame(tmpdatapack, columns=arr_col)
    df.to_csv("data_for_smilecomp_bace.csv", index=False)
    f.close()


load_data()
