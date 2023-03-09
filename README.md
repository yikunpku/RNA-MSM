# RNA-MSM
![RNA-MSM_Fig1](https://user-images.githubusercontent.com/122002181/224082314-026873db-56d0-42cb-8930-3249b553a924.png)

# Description
This repository contains codes and pre-trained weights for MSA RNA language model(**RNA-MSM**) as well as RNA secondary structure 
and solvent accessibility task. 

**RNA-MSM** is the first unsupervised MSA RNA language model based on aligned homologous sequences that outputs both embedding 
and attention map to match different types of downstream tasks. 

The resulting RNA-MSM model produced attention maps and embeddings that have direct correlations to RNA secondary structure 
and solvent accessibility without supervised training. Further supervised training led to predicted secondary structure and 
solvent accessibility that are **significantly more accurate than current state-of-the-art techniques**. Unlike many previous studies, 
we would like to emphasize that we were **extremely careful in avoiding over training**, a significant problem in applying deep learning 
to RNA by **choosing validation and test sets structurally different from the training set**.
