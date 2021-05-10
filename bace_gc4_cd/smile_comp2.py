"""
This python script reads in a SMILES string and then finds the maximum common substructure
for that SMILES string with data that is also passed in (can accept either <STRING>   or <NAME,STRING>)

Data is assumed to be a csv with the following format: <PDBID>,<SMILE>,<LIG_ID>
"""

from rdkit import Chem
from rdkit.Chem import rdFMCS
import pandas as pd

refdata = pd.read_csv("data_for_smilecomp.csv")  # PDBID	SMILE	LIG_ID	affinity

gc4_label_ori = pd.read_csv("BACE_score_compounds_D3R_GC4_answers.csv")
gc4_label = gc4_label_ori

out = open('similar_pdbid_info2.tsv','w')

for i, Cmpd_ID in enumerate(gc4_label['Cmpd_ID']):
    smile = gc4_label['SMILES'][i]
    base = Chem.MolFromSmiles(smile)
    best_pdb = ''
    best_ligid = ''
    best_smile = ''
    best_num_bonds = 0  # indicator for the num of bonds in the max substructure.
    best_smarts = ''

    # find the lig in Pocketome's entry for BACE1 with the maximum common substructure to the given SMILE
    for i, comp_smile in enumerate(refdata['SMILE']):

        comp = Chem.MolFromSmiles(comp_smile)
        if not comp: # some ligand is not appropriate
            continue
        res = rdFMCS.FindMCS([base, comp], completeRingsOnly = False, bondCompare = rdFMCS.BondCompare.CompareOrder)

        if res.numBonds > best_num_bonds:
            best_num_bonds = res.numBonds
            best_pdb = refdata['PDBID'][i]
            best_ligid = refdata['LIG_ID'][i]
            best_smile = comp_smile
            best_smarts = res.smartsString

    # now we have the best pdb id for the lig with the largest common substructure to our ligand.
    out.write(Cmpd_ID + '\t' + smile + '\t' + best_pdb + '\t' + best_ligid + '\t' + best_smile + '\t' + best_smarts + '\n')
out.close()
