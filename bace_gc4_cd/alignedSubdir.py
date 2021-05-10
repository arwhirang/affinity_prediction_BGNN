import copy
import sys
import os
from os import walk
import subprocess
from spyrmsd import io
from spyrmsd import spyrmsd

DATAPATH = "conf"
NEW_SUB_PATH = "alignedBACEsubset"
TRUTHPATH = "truth"
SEP_IDX=5


def aligned2sub(path, filename, new_sub_path):
    fi = open(path + "/" + filename)
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

    new_dir = filename[SEP_IDX:].replace("_aligned.sdf", "")
    num_prev_files = 0
    if not os.path.exists(new_sub_path):
        os.mkdir(new_sub_path)
    if not os.path.exists(new_sub_path + "/" + new_dir):
        os.mkdir(new_sub_path + "/" + new_dir)
    else:
        num_prev_files = len([_ for _ in os.listdir(new_sub_path + "/" + new_dir)])

    for i in range(tmpcnt):
        new_fname = str(num_prev_files + i)
        fw = open(new_sub_path + "/" + new_dir + "/" + new_fname + ".sdf", "w")
        for aline in dicFile[i]:
            fw.write(aline)
        fw.write("$$$$\n")
        fw.close()


# main
fnames = []
for (dirpath, dirnames, _filename) in walk(DATAPATH):
    fnames.extend(_filename)
    break

forcsv = {}
for _fname in fnames:
    if _fname.endswith("_aligned.sdf"):
        aligned2sub(DATAPATH, _fname, NEW_SUB_PATH)


