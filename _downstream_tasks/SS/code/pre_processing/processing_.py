import sys
sys.path.append("/lustre/home/mlang/2D_Structure_ENV/lib/python3.9/site-packages")
sys.path.append("/lustre/software/anaconda/anaconda3-2019.10-py37/lib/python3.7/site-packages/")
import numpy as np
from Bio import SeqIO
import re
from pathlib import Path

def get_chain_sequence(fasta_file):
    """
    input :fasta file name
    return a list contain the pdb_chain PDB ID and chain name and sequence information of a chain
    """
    # fasta_file=FASTA_FILE
    chain_informations = []
    for record in SeqIO.parse(fasta_file, 'fasta'):
        description = record.description
        pdb_id = description.split('|')[0].split("_")[0]
        pdb_id_num = record.id.split("|")[0]
        if "," in pdb_id:
            pdb_id = pdb_id.split(",")[0]
        # print(description)
        if 'auth' not in description:
            chain_id = description.split('|')[1].split(" ")[1]
            pdb_chain = pdb_id + "_" + chain_id
        else:
            qurey = re.search(r"(?<=\[).*?(?=\])", description)
            chain_id = qurey.group().split(" ")[1]
            pdb_chain = pdb_id + "_" + chain_id
        if "," in pdb_chain:
            pdb_chain = pdb_chain.split(",")[0]
        sequence = str(record.seq)
        sequence_chain = (pdb_id_num, pdb_chain, sequence)
        chain_informations.append(sequence_chain)
    return chain_informations

def load_am(fasta_file,am_file):
    chain_informations = get_chain_sequence(fasta_file)
    am_all = []
    for pdb_id_num, pdb_chain, sequence in chain_informations:
        #fin =  am_file + '{}.a2m_atp_msatrans.npy'.format(str(pdb_id_num))
        fin = am_file + '{}.npy'.format(str(pdb_id_num))
        fin_ts2 = am_file + '{}.npy'.format(str(pdb_chain))
        #fin = am_file + '{}.npy'.format(str(pdb_id_num))
        #fin_ts2 = am_file + '{}.npy'.format(str(pdb_chain))
        if Path(fin).is_file():
            with open(fin, "rb") as f:
               am_tmp = []
               am = np.load(f)
               print("ori",am.shape)
               #am = am.reshape(am.shape[0]*am.shape[1],am.shape[2],am.shape[3])
               #am = am[:,1:-1,1:-1]
               #print("shape remove start and end:", am.shape)
               am_tmp.append(pdb_chain)
               am_tmp.append(am)
               am_all.append(am_tmp)
        if Path(fin_ts2).is_file():
            with open(fin_ts2, "rb") as f:
               print("ts2 chain:",pdb_chain)
               am_tmp = []
               am = np.load(f)
               print(am.shape)
               #am = am.reshape(am.shape[0]*am.shape[1],am.shape[2],am.shape[3])
               am_tmp.append(pdb_chain)
               am_tmp.append(am)
               am_all.append(am_tmp)
    return am_all
train_fasta = "./2d_structure_sequence_min32_max1k.fasta"
am_file = "./reducted_test/embedding/"
am_all = load_am(train_fasta,am_file)
np.save("embedding_reducted.npy", am_all)

