# Blind affinity prediction using BGNN and transfer learning on re-docking dataset  
Under review for Bioinformatics

## Preparation(pose prediction, too)  
conda environment should have rdkit/obfit(open babel)  
  
1 PDBbind data (re-docking)    
- Download PDBbind data(v2019) both "general set" and "refined set" and merge all the files inside folder pdbbind_files.  
- run "utilities/pdbbind_redo.py" to create re-docking dataset from PDBbind data  
- The pdbbind_redo.py file contains python path for the conda environment. Please modify the PATH_TO_PYTHON accordingly.  

2 chembl_bace data (cross-docking, "bace_chembl_cd")  
- One can download IC50 values of the target protein BACE (chembl id:CHEMBL4822) or use the BACE_IC50.csv file to start  
- run "make_data_smilecomp.py", "smile_comp2.py", "conformer_gen_BACE.py", and "alignedSubdir.py" in this order.  
- Some of the files contains python path for the conda environment. Please modify the PATH_TO_PYTHON accordingly.  
- move the "alignedsubset" directory to the "src2" and change the name to "chembl_bace"  
  
3 BACE data (cross-docking, "bace_gc4_cd")  
- Start with the BACE_score_compounds_D3R_GC4_answers.csv file  
- run "make_data_smilecomp.py", "smile_comp_bace.py", "conformer_generation_bace.py", and "alignedSubdir_bace.py" in this order.  
- Some of the files contains python path for the conda environment. Please modify the PATH_TO_PYTHON accordingly.  
- move the "alignedBACEsubset" directory to the "src2" and change the name to "BACE"  
 
## data and source code path required to run BGNN library  
- pdbbind_files ==> Preparatory work is required (too big to upload all files)  
- pdbbind_index ==> no preparation required  
- src2/chembl_bace ==> Preparatory work is required (too big to upload all files)  
- src2/BACE ==> Preparatory work is required (too big to upload all files)  
- src2/CATS ==> unzip the compressed file  
- src2/gc3_CATS ==> unzip the compressed file  
- The name "src2" means that the code is run on the 2nd gpu card
  
## affinity prediction  
- run the "src2/train.py"  
  
I am beutifying the codes... please ask me anything if you feel confused.
