# Blind affinity prediction using BGNN and transfer learning on re-docking dataset  
Under review for Bioinformatics

## Preparation  
1 PDBbind data  
- Download PDBbind data(v2019) both "general set" and "refined set" and merge them.  
- run "" to create re-docking dataset from PDBbind data  

## data and source code path for our library  
- pdbbind_files  
- pdbbind_index  
- src  
- src/chembl  
- src/bace  
- src/cats

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
  
