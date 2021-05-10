'''
This python script reads in a SMILES string and then finds the maximum common substructure
for that SMILES string with data that is also passed in (can accept either <STRING>   or <NAME,STRING>)

Data is assumed to be a csv with the following format: <PDBID>,<SMILE>,<LIG_ID>
'''

from rdkit import Chem
from rdkit.Chem import rdFMCS
import pandas as pd

refdata = pd.read_csv("data_for_smilecomp_bace.csv")#PDBID	SMILE	LIG_ID	affinity

base_dat = pd.read_csv("BACE_IC50.csv")

out = open('similar_pdbid_info_bace.tsv','w')
for i, y in enumerate(base_dat['Molecule ChEMBL ID']):
    smile = base_dat['Smiles'][i]
    base = Chem.MolFromSmiles(smile)
    best_pdb = ''
    best_ligid = ''
    best_smile = ''
    best_num_bonds = 0 # indicator for the num of bonds in the max substructure.
    best_smarts = ''

    # find the lig in Pocketome's entry for BACE1 with the maximum common substructure to the given SMILE
    for j, comp_smile in enumerate(refdata['SMILE']):

        comp = Chem.MolFromSmiles(comp_smile)
        if not comp: # some ligand is not appropriate
            continue
        res = rdFMCS.FindMCS([base, comp], completeRingsOnly = False, bondCompare = rdFMCS.BondCompare.CompareOrder,
                             timeout=10)

        if res.numBonds > best_num_bonds:
            best_num_bonds = res.numBonds
            best_pdb = refdata['PDBID'][j]
            best_ligid = refdata['LIG_ID'][j]
            best_smile = comp_smile
            best_smarts = res.smartsString

    # now we have the best pdb id for the lig with the largest common substructure to our ligand.
    out.write(y + "_" + str(i) + '\t' + smile + '\t' + best_pdb + '\t' + best_ligid + '\t' + best_smile + '\t' + best_smarts + '\n')
out.close()
