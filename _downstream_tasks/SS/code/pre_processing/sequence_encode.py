#from Bio import SeqIO
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
def one_hot_encode(sequences):
    sequences = sequences
    sequences_arry = np.array(list(sequences)).reshape(-1, 1)
    lable = np.array(list('ACGU')).reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(lable)
    seq_encode = enc.transform(sequences_arry).toarray()
    print(seq_encode.shape)
    return (seq_encode)
