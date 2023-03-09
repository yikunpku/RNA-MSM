import numpy as np


def metric(pred_data, pred_path):
    """
    INPUT: a tuple of (predict, label)
    output prob_file
    """
    pred_label = pred_data
    for id, y_pred in pred_label:
        y_pred = y_pred.cpu().numpy()  # GPU data to CPU
        y_pred = np.squeeze(y_pred)
        pred_file = pred_path + "{}.prob".format(id)
        np.savetxt(pred_file, y_pred)  # .numpy())
