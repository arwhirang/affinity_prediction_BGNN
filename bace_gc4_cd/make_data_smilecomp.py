import pandas as pd
import os
from rdkit import Chem
import math
import numpy as np
from shutil import copyfile


#def uM_to_kcal(ic50):
#    #convert the ic50 values to binding affinity
#    return math.log(ic50 * 1e-6) * 0.5961


def extract_pdbid_from_name(target):
    retlist = []
    f = open('pdbbind_index/INDEX_all_name.2019')
    for line in f.readlines():
        splitted = line.strip().split('  ')
        pdbid = splitted[0]
        pro_id = splitted[2]
        if pro_id == target:
            retlist.append(pdbid)
    f.close()
    return retlist
    

def load_data():
    target_pdbs = extract_pdbid_from_name("P10275")  # P10275 :Androgen
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

    f = open('pdbbind_index/INDEX_all.2019')
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

        if not os.path.exists("pdbbind_files/" + pdbid):  # some pdbid only exists in the index files
            print("pdbid not found")
            continue
            
        sdfMOLs = Chem.MolFromMolFile("pdbbind_files/" + pdbid + "/" + pdbid + "_ligand.sdf", sanitize=False)
        #sdfMOLs = Chem.SDMolSupplier("pdbbind_files/" + pdbid + "/" + pdbid + "_ligand.sdf")[0]
        if not sdfMOLs:
            print("not a valid mol-ligand1:", pdbid)
            continue

        pvalue = -np.log10(value) + 6
            
        # for truth ligand/protein file for conformer generation processes
        copyfile("pdbbind_files/" + pdbid + "/" + pdbid + "_ligand.sdf", "truth/" + pdbid + "_ligand.sdf")
        copyfile("pdbbind_files/" + pdbid + "/" + pdbid + "_protein.pdb", "truth/" + pdbid + "_protein.pdb")
            
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
    df.to_csv("data_for_smilecomp_ar.csv", index=False)
    f.close()
    
load_data()
