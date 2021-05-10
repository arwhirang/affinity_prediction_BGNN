import os
from rdkit import Chem
from rdkit.Chem import rdFMCS
import subprocess
import copy
import pandas as pd

PATH_TO_PYTHON = "/home/sysgen/.conda/envs/cuda102/bin/python"

pathToProjHome = "../"


def gen_aligned():
    # This func iterate every pdbs, and generate re-docking dataset
    # 1. generated conformers will locate inside "subconfs" subdir (smis, conf.sdfs, aligned.sdfs - used this only)
    # 2. separated aligned sdfs will locate inside "sep_sdfs" dir
    # sometimes, rdkit or obfit could not handle the given sdf (ligand), then no conformers are generated
    # the pose prediction is retrieved from GNINA participants
    # https://drugdesigndata.org/php/d3r/gc4/combined/scoringboth/index.php?component=1479&method=combined

    f = open(pathToProjHome + 'pdbbind_index/INDEX_all.2019')
    for line in f.readlines():
        if line[0] == '#':
            continue

        lines = line.split('/')[0].strip().split('  ')
        pdbid = lines[0]

        # read sdf file
        if not os.path.exists(pathToProjHome + "pdbbind_files/" + pdbid):  # some pdbid only exists in the index files
            continue

        # skip files that are already processed
        if os.path.exists(pathToProjHome + "pdbbind_files/" + pdbid + "/sep_sdfs"):
            continue

        sdfMOLs = Chem.MolFromMolFile(pathToProjHome + "pdbbind_files/" + pdbid + "/" + pdbid + "_ligand.sdf", sanitize=False)
        sdfMOLs.UpdatePropertyCache(strict=False)
        if not os.path.exists(pathToProjHome + "pdbbind_files/" + pdbid + "/subconfs"):
            os.mkdir(pathToProjHome + "pdbbind_files/" + pdbid + "/subconfs")

        # set variables
        smi_base = Chem.MolToSmiles(sdfMOLs)
        base_mol = Chem.MolFromSmiles(smi_base)
        smi_ref = Chem.MolToSmiles(sdfMOLs, isomericSmiles=False)
        ref_mol = Chem.MolFromSmiles(smi_ref)
        if not base_mol or base_mol is None or not ref_mol or ref_mol is None:
            continue

        # calc smarts
        res = rdFMCS.FindMCS([base_mol, ref_mol], completeRingsOnly=False, bondCompare=rdFMCS.BondCompare.CompareOrder)

        # gen conformers
        smi_name = pathToProjHome + "pdbbind_files/" + pdbid + "/subconfs/" + pdbid + '.smi'
        outname = pathToProjHome + "pdbbind_files/" + pdbid + "/subconfs/" + pdbid + "_conformers.sdf"

        # writing file for rdconf.py script
        with open(smi_name, 'w') as out:
            out.write(smi_base)

        # call rdconf.py
        subprocess.check_call(PATH_TO_PYTHON + ' rdconf.py --maxconf 30 ' + smi_name + ' ' + outname, shell=True)

        # apply obfit
        sent = "obfit '" + res.smartsString + "' " + pathToProjHome + "pdbbind_files/" + pdbid + "/" + pdbid + "_ligand.sdf" + ' ' \
               + outname + " > " + pathToProjHome + "pdbbind_files/" + pdbid + "/subconfs/" + pdbid + "_aligned.sdf"
        try:
            subprocess.check_call(sent, shell=True)
        except:
            print("error command:", sent)
            continue

        # separate the *aligned.sdf into discrete sdf (usually less than maxconf)
        # store each sdfs and save them to dict
        fi = open(pathToProjHome + "pdbbind_files/" + pdbid + "/subconfs/" + pdbid + "_aligned.sdf")
        lines = fi.readlines()
        dicFile = {}
        tmplines = []
        tmpcnt = 0
        for aline in lines:
            if aline.startswith("$$$$"):
                dicFile[tmpcnt] = copy.deepcopy(tmplines)
                tmplines = []
                tmpcnt += 1
                continue
            tmplines.append(aline)
        fi.close()
        # make directory for storing separated sdf
        new_sub_path = pathToProjHome + "pdbbind_files/" + pdbid + "/sep_sdfs"
        if not os.path.exists(new_sub_path):
            os.mkdir(new_sub_path)
        # write sdf to the pre-defined path
        for i in range(tmpcnt):
            fw = open(new_sub_path + "/" + pdbid + str(i) + ".sdf", "w")
            for aline in dicFile[i]:
                fw.write(aline)
            fw.write("$$$$\n")
            fw.close()
        # Separation finished


gen_aligned()
