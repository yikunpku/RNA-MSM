import torch
from pathlib import Path
from argparse import ArgumentParser
import os
from code.pre_processing.data_fomat import format_input_shape
from code.pre_processing.data_processing import *
from code.model import renet_b16
from code.post_processing.processing_output import prob_to_secondary_structure

current_directory = Path(__file__).parent.absolute()

def predict(model, data_use_test, device):
    """
    model test from load the checkpoint file
    """
    model.eval()
    pred_label = []
    with torch.no_grad():
        for test_data in data_use_test:
            rna_name, seq, x_test = format_input_shape(test_data, device)
            pred = model(x_test)
            pred = torch.sigmoid(pred)
            pred = torch.squeeze(pred)
            pred_label.append((rna_name, seq, pred))
    return pred_label

def getParam():
    parser = ArgumentParser()
    parser.add_argument('--rootdir', default=current_directory,
                        type=str)
    parser.add_argument('--featdir', default='./results',
                        type=str)
    parser.add_argument('--rnaid', default='2DRB_1',
                        type=str)
    parser.add_argument('--plots', default=False, type=bool,
                        help='Set this to "True" to get the 2D plots of predicted secondary structure by RNA-MSM-SS; default = False',
                        metavar='')
    parser.add_argument('--device', default='cpu',
                    type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = getParam()
    
    inputs_fasta = os.path.join(args.featdir, args.rnaid + ".fasta")
    model_path = os.path.join(args.rootdir, "model/rna-msm_attention.pt")
    inputs_attention = os.path.join(args.featdir, args.rnaid + "_atp.npy") 
  
    l_feature_obj = DataProcess(inputs_fasta,inputs_attention)
    test_data = l_feature_obj.feature_load()
    model = renet_b16().to(args.device)
    model_para = torch.load(model_path, map_location=torch.device(args.device))
    model.load_state_dict(model_para)
    preds = predict(model, test_data, args.device)
    for rna_name, seq, model_output in preds:
        prob_to_secondary_structure(model_output, seq, rna_name, args, args.rootdir, args.featdir)  # model_outputs, seq, name, args, base_path
