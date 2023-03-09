#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Title     : RSA_predictor.py
# Created by: julse@qq.com
# Created on: 2022/12/7 14:17
import glob
from pathlib import Path
import pickle
import random
import re
import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import torch
from Bio import SeqIO
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import sys
current_directory = Path(__file__).parent.absolute()
sys.path.append(str(current_directory))

class bcolors:
    RED   = "\033[1;31m"
    BLUE  = "\033[1;34m"
    CYAN  = "\033[1;36m"
    GREEN = "\033[0;32m"
    RESET = "\033[0;0m"
    BOLD    = "\033[;1m"
    REVERSE = "\033[;7m"

def check_path(dirout,file=False):
    if file:dirout = dirout.rsplit('/',1)[0]
    try:
        if not os.path.exists(dirout):
            print('make dir '+dirout)
            os.makedirs(dirout)
    except:
        print(f'{dirout} have been made by other process')


def doSavePredict_single(_id,seq,predict_rsa,fout,des,pred_asa=None):
    check_path(fout)
    BASES = 'AUCG'
    asa_std = [400, 350, 350, 400]
    dict_rnam1_ASA = dict(zip(BASES, asa_std))
    sequence = re.sub(r"[T]", "U", ''.join(seq))
    sequence = re.sub(r"[^AGCU]", BASES[random.randint(0, 3)], sequence)
    ASA_scale = np.array([dict_rnam1_ASA[i] for i in sequence])

    if pred_asa is None:
        pred_asa = np.multiply(predict_rsa, ASA_scale).T
    else:
        predict_rsa = pred_asa/ASA_scale
    col1 = np.array([i + 1 for i, I in enumerate(seq)])[None, :]
    col2 = np.array([I for i, I in enumerate(seq)])[None, :]
    col3 = pred_asa
    col4 = predict_rsa
    if len(col3[col3 == 0]):
        exit(f'error in predict\t {_id},{seq}')
    temp = np.vstack((np.char.mod('%d', col1), col2, np.char.mod('%.2f', col3), np.char.mod('%.3f', col4))).T
    if fout:np.savetxt(fout + f'{_id}.txt', (temp), delimiter='\t\t', fmt="%s",
               header=f'#{des}\n#index\t\tnt\t\tASA\t\tRSA\n',
               comments='')

    return pred_asa,predict_rsa

def one_hot_encode(sequences,alpha='ACGU'):
    print(sequences)
    sequences_arry = np.array(list(sequences)).reshape(-1, 1)
    lable = np.array(list(alpha)).reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(lable)
    seq_encode = enc.transform(sequences_arry).toarray()
    return (seq_encode)

def plot(pred_asa,label='RNA-MSM',title='predicted RSA'):
    plt.title(title)
    plt.xlabel('Nucleotide index')
    plt.ylabel(title.split(' ')[-1]+' value')
    plt.plot(pred_asa,'-.',marker='*',label=label)
    # plt.plot(pred_asa,'-.',label=label)
def seed_everything(seed=2022):
    print('seed_everything to ',seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def getParam():
    parser = ArgumentParser()
    # data
    parser.add_argument('--rootdir', default=current_directory,
                        type=str)
    parser.add_argument('--featdir', default='/mnt/d/_Codes/_Pythonprojects/RNA-MSM-republic/results',
                        type=str)
    parser.add_argument('--rnaid', default='2DRB_1',
                        type=str)
    parser.add_argument('--device', default='cpu',
                        type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = getParam()
    fasta = os.path.join(args.featdir, args.rnaid + ".fasta")
    ffeat = os.path.join(args.featdir, args.rnaid + "_emb.npy")
    fmodel = os.path.join(args.rootdir, 'models/OH+RNA-MSM_Emb/model_pcc_*.pt')
    fstatic = os.path.join(args.rootdir, 'models/OH+RNA-MSM_Emb/statistic_dict_{}.pickle')
    device = args.device
    out = os.path.join(args.featdir, 'RSA_result')
    check_path(out)

    seed_everything(seed=2022)
    models = [(model_path, torch.load(model_path, map_location=torch.device(device))) for model_path in
              glob.glob(fmodel)]
    print(bcolors.BLUE,f'loading {len(models)} models from {fmodel} pattern',bcolors.RESET)

    statis_dict = pickle.load(open(fstatic.format('oh'), 'rb'))
    mu_oh, std_oh = statis_dict['mu'], statis_dict['std']
    statis_dict = pickle.load(open(fstatic.format('emb'), 'rb'))
    mu_emb, std_emb = statis_dict['mu'], statis_dict['std']

    with torch.no_grad():
        for pdbid,seq in [(record.id,record.seq) for record in SeqIO.parse(fasta,'fasta')]:

            emb = np.load(ffeat.format_map({'pdbid': pdbid}))
            emb = (emb - mu_emb) / std_emb

            oh = one_hot_encode(seq)
            oh = (oh - mu_oh) / std_oh

            mask = np.ones((emb.shape[0], 1))
            x_train = np.concatenate([oh, emb, mask], axis=1)
            x_train = np.expand_dims(x_train,0)
            x_train = torch.from_numpy(x_train).transpose(-1,-2)
            x_train = x_train.to(device, dtype=torch.float)
            ensomle_asa_ours = []
            print(f'predict {pdbid} with {len(seq)} nts')
            for i,(model_path,model) in enumerate(models):
                model.eval()
                outputs = model(x_train)
                out_pred = torch.squeeze(outputs)
                dirout = out+f'/model_{i}/'
                check_path(dirout)

                pred_asa,predict_rsa  = doSavePredict_single(pdbid, seq, out_pred.numpy(),
                              dirout, f'{pdbid} predict by {model_path.rsplit("/",1)[1]}\n')
                ensomle_asa_ours.append(pred_asa)

            dirout = out+f'/ensemble/'
            check_path(dirout)
            ensomle_asa_ours = np.array(ensomle_asa_ours).mean(0)
            pred_asa,predict_rsa = doSavePredict_single(pdbid, seq, None,
                                            dirout, f'{pdbid} predict by ensemble model\n',pred_asa=ensomle_asa_ours)

            plot(predict_rsa, label='RNA-MSM RSA predictor')
            plt.legend()
            plt.savefig(out+f'/{pdbid}_rsa.png')

            plt.clf()
            plot(pred_asa,label='RNA-MSM ASA predictor',title='predicted ASA')
            plt.legend()
            plt.savefig(out+f'/{pdbid}_asa.png')









