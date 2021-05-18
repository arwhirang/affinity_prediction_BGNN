# Blind affinity prediction using BGNN and transfer learning on re-docking dataset  
Under review for MDPI IJMS  

## Preparation(pose prediction, too)  
conda environment should have rdkit/obfit(open babel)/sklearn/pandas/scipy/pytorch  
    
1 PDBbind data (re-docking)  
- Download PDBbind data(v2019) both "general set" and "refined set" and merge all the files inside folder pdbbind_files.  
- run "utilities/pdbbind_redo.py" to create re-docking dataset from PDBbind data  
- The pdbbind_redo.py file contains python path for the conda environment. Please modify the PATH_TO_PYTHON accordingly.  
- Assuming that it takes up to an hour to process 100 instances, the job will finish in less than 40 hours.  
  
2 chembl_bace data (cross-docking, "bace_chembl_cd")  
- One can download IC50 values of the target protein BACE (chembl id:CHEMBL4822) or use the "BACE_IC50.csv" file to start  
- run "make_data_smilecomp.py", "smile_comp2.py", "conformer_gen_BACE.py", and "alignedSubdir.py" in this order.  
- Some of the files contains python path for the conda environment. Please modify the PATH_TO_PYTHON accordingly.  
- copy all the contents inside the "truth" folder into "alignedsubset"  
- copy "BACE_IC50.csv" into "alignedsubset"  
- copy "similar_pdbid_info_bace.tsv" into "alignedsubset"  
- move the "alignedsubset" directory to the "src2" and change the name to "chembl_bace"  
  
3 BACE data (cross-docking, "bace_gc4_cd")  
- Start with the "BACE_score_compounds_D3R_GC4_answers.csv" file  
- run "make_data_smilecomp.py", "smile_comp_bace.py", "conformer_generation_bace.py", and "alignedSubdir_bace.py" in this order.  
- Some of the files contains python path for the conda environment. Please modify the PATH_TO_PYTHON accordingly.  
- copy all the contents inside the "truth" folder into "alignedBACEsubset"  
- copy "BACE_score_compounds_D3R_GC4_answers.csv" into "alignedBACEsubset"  
- copy "similar_pdbid_info2.tsv" into "alignedBACEsubset"  
- move the "alignedBACEsubset" directory to the "src2" and change the name to "BACE"  
 
## data and source code path required to run BGNN library  
- pdbbind_files ==> Preparatory work is required (too big to upload all files)  
- pdbbind_index ==> no preparation required  
- src2/chembl_bace ==> Preparatory work is required (too big to upload all files)  
- src2/BACE ==> Preparatory work is required (too big to upload all files)  
- src2/CATS ==> unzip the compressed file  
- src2/gc3_CATS ==> unzip the compressed file  
- The name "src2" means that the code is run on the 2nd gpu card
  
## affinity prediction(BGNN)  
- run the "src2/train.py"  
- In the first run, modify the hyperparameter usePickledData to False.  
    
## ensemble  
- I used ensemble for the final result. the code can be found in the utilities folder  
  
  
please ask me anything if you feel confused.
