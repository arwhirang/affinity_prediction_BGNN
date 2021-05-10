import copy
import sys
import os
from os import walk


DATAPATH = "conf"
NEW_SUB_PATH = "alignedsubset"


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

    new_dir = filename.split("_")[1]
    if not os.path.exists(new_sub_path + "/" + new_dir):
        os.mkdir(new_sub_path + "/" + new_dir)

    for i in range(tmpcnt):
        fw = open(new_sub_path + "/" + new_dir + "/" + str(i) + ".sdf", "w")
        for aline in dicFile[i]:
            fw.write(aline)
        fw.write("$$$$\n")
        fw.close()


# main
fnames = []
for (dirpath, dirnames, _filename) in walk(DATAPATH):
    fnames.extend(_filename)
    break

if not os.path.exists(NEW_SUB_PATH):
    os.mkdir(NEW_SUB_PATH)

forcsv = {}
for _fname in fnames:
    if _fname.endswith("_aligned.sdf"):
        aligned2sub(DATAPATH, _fname, NEW_SUB_PATH)


