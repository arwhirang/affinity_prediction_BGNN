# Blind affinity prediction using BGNN and transfer learning on re-docking dataset  
Under review for Bioinformatics

## Preparation  
1 PDBbind data (re-docking)    
- Download PDBbind data(v2019) both "general set" and "refined set" and merge all the files inside folder pdbbind_files.  
- run "utilities/pdbbind_redo.py" to create re-docking dataset from PDBbind data  
- The pdbbind_redo.py file contains python path for the conda environment. Please modify the PATH_TO_PYTHON accordingly.

2 chembl_bace data (cross-docking)  
- One can download IC50 values of the target protein BACE (chembl id:CHEMBL4822) or use the BACE_IC50.csv file to start 
 
3 BACE data (cross-docking)  
- Start with the BACE_score_compounds_D3R_GC4_answers.csv file
 
## data and source code path required to run BGNN library  
- pdbbind_files ==> Preparatory work is required (too big to upload all files)  
- pdbbind_index ==> no preparation required  
- src2/chembl_bace ==> Preparatory work is required (too big to upload all files)  
- src2/BACE ==> Preparatory work is required (too big to upload all files)  
- src2/CATS ==> unzip the compressed file  
- src2/gc3_CATS ==> unzip the compressed file  

## pose prediction  
1   
2 Generate conformers 
3 Find a reference ligand  
4 Superimpose on the reference ligand 
5 Store the data for mini-batch 
  
## affinity prediction  
1 Apply ECIF  
2 Character embedding on each atoms 
3 Self-attention 
4 Bipartite graph neural network 
5 result 
  
I am beutifying the codes... please ask me anything if you feel confused.
