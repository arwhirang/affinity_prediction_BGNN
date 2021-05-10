import os
import pandas as pd
import subprocess
from rdkit import Chem
import glob
import numpy as np

PATH_TO_PYTHON = "/home/sysgen/anaconda3/envs/cuda102/bin/python"

ref_data = pd.read_csv('similar_pdbid_info2.tsv', header = None, names = ['d3r_id','d3r_smile','pdb_id','pdb_lig','pdb_smile','smarts'], sep='\t')

for _, row in ref_data.iterrows():
    if row['d3r_id'] == "BACE_" + row['pdb_lig']:
        continue

    # ============code for rdconf====================
    if not os.path.exists("conf"):
        os.mkdir("conf/")

    fname = 'conf/' + row['d3r_id'] + '.smi'
    outname = 'conf/' + row['d3r_id']+"_conformers.sdf"
    
    # writing file for rdconf.py script
    with open(fname, 'w') as out:
        out.write(row['d3r_smile'])
    
    subprocess.check_call(PATH_TO_PYTHON + ' rdconf.py --maxconf 50 ' + fname + ' ' + outname, shell=True)#--maxconf 30
    # ============code for rdconf====================
    
    # ============code for obfit=====================
    if row['pdb_id'] == 'BACE':
        ref = 'truth/bace' + str(row['pdb_lig']) + '.sdf'
    else:
        ref = 'truth/' + str(row['pdb_id']) + "_ligand.sdf"
        
    sent="obfit '" + row['smarts'] + "' " + ref + ' ' + outname + ' > conf/' + row['d3r_id'] + "_aligned.sdf"
    try:
        subprocess.check_call(sent, shell=True)
    except:
        print (row['d3r_id'], sent)
    # ============code for obfit=====================

#Generate solo-conformers
conf = glob.glob('conf/*conformers.sdf')
for c in conf:
    newname = c.split('_conformers')[0] + '_sconf.sdf'
    with open(newname,'w') as out:
        for line in open(c,'r').readlines():
            out.write(line)
            if '$$$$' in line:
                break
