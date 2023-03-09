import torch
import numpy as np
import pickle


def load_pickle_file(file):
    """
    input: pickle file
    return:list of feature, [(),(),().....()], in every tuple contain five feature:
    pdb_chain: PDB ID and Chain ID for example :4U3M_A
    pairwise_seq_feat: one-hot encoding of sequence, shape:[seq_len,seq_len,8]
    pairwise_embedding: language model output, shape:[seq_lem,seq_len,2048]
    label: golden truth of RNA second structure map, shape: [seq_len,seq_len]
    missing_nt_index: index of the residue of the no structure in pdb file
    """
    with open(file, "rb") as f:
        data = pickle.load(f)
        return data


def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)


def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized


def format_input_shape(model_input, device):
    """
    input : lsit of feature
    return: model input and the missing index that use to mask in loss compute
            x_train: model feature input, shape:[1,channel,height, width]
            y_train: golden true [heigth,width]
            missing_nts: missing residue index
            all return data is torch format data
    model_input.append(pdb_chain)
    model_input.append(label)
    model_input.append(missing_nt_index)
    #model_input.append(single_feature)
    model_input.append(pairwise_feature)
    """
    pdb_chain, seq, attentionmap = model_input[0], model_input[1], model_input[2]
    x = attentionmap
    x = np.expand_dims(x, axis=0)  # [L,L,2048]->[1,L,L,2048],batch_size=1
    print("attention_map + oh shape:",x.shape)
    x = np.transpose(x, (0, 3, 1, 2))
    x = torch.from_numpy(x)
    x = x.to(device, dtype=torch.float)  # (device="cuda:0", memory_format=torch.channels_first)
    return pdb_chain,seq, x,
