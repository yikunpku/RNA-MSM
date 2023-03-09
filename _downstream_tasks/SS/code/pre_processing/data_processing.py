import numpy as np
from Bio import SeqIO
from code.pre_processing.sequence_encode import one_hot_encode
from code.pre_processing.outer_concatenation import outer_concatenation, matrix_concatenation
from pathlib import Path


class DataProcess(object):
    def __init__(self,
                 fasta_file,
                 attention_file,
                 ):
        self.fasta_file = fasta_file
        self.attention_file = attention_file

    def load_am(self, fasta_id):
        f_attention = self.attention_file
        #print("fasta_id and f_attention",fasta_id,f_attention)
        if Path(f_attention).is_file():
            with open(f_attention, "rb") as f:
                am = np.load(f)
                am = am.transpose(1,2,0)
                return am
        else:
            print("attention map is not exists,please extract attention")

    def process_onehot_feature(self, sequence):
        onehot_feature = one_hot_encode(sequence)
        pairwise_onehot = outer_concatenation(onehot_feature, onehot_feature)
        return pairwise_onehot

    def feature_load(self):
        """
        Input: a series of file path
        :return: [pdb_chain, label, missing_nt_index,(feature)] and feature:(single_feature,pairwise_feature)
        """
        all_input = []
        fasta_file = self.fasta_file
        for record in SeqIO.parse(fasta_file, 'fasta'):
            fasta_id = record.description
            seq = str(record.seq)
            seq_oh = self.process_onehot_feature(seq) # [L,L,8]
            #print("seq_oh shape",seq_oh.shape)
            seq_attention = self.load_am(fasta_id)  # [L,L,120]
            #print("seq_attention",seq_attention.shape)
            onehot_attention = np.concatenate((seq_oh, seq_attention), axis=2)
            all_input.append([fasta_id, seq, onehot_attention])
        return all_input
