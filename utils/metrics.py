from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from .tensor import coerce_numpy
from sklearn.metrics import auc

@coerce_numpy
def rna_compute_precisions(
    validation_step_outputs: torch.Tensor,
    minsep: int = 0,
    step: int = 0.001,
):
    T = np.arange(0, 1 + step, step)[None, :]  # threshold value [1,1001]
    tp = 0; fn = 0; fp = 0; tn = 0

    for batch in validation_step_outputs:
        predictions = batch["predictions"]
        targets = batch["tgt"]
        missing_nt_index = batch["missing_nt_index"]
        # Check sizes
        if predictions.size() != targets.size():
            raise ValueError(
                f"Size mismatch. Received {rnaid} predictions of size {predictions.size()}, "
                f"targets of size {targets.size()}"
            )

        if predictions.dim() == 3:
            predictions = predictions.squeeze()     # [BS, L, L]-> [L,L]
        if targets.dim() == 3:
            targets = targets.squeeze()     # [BS, L, L]-> [L,L]

        seqlen, _ = predictions.size()
        device = predictions.device

        mask = torch.triu(torch.ones((seqlen, seqlen), device=device), minsep+1) > 0    # 返回上三角矩阵，对角线偏移量1
        if missing_nt_index is None:
            targets = targets.masked_select(mask)
            predictions = predictions.masked_select(mask)
        else:
            for i in missing_nt_index:
                mask[i, :] = 0
                mask[:, i] = 0
            targets = targets.masked_select(mask)
            predictions = predictions.masked_select(mask)

        targets = targets.unsqueeze(1).cpu().numpy()    #[n,1]
        predictions = predictions.unsqueeze(1).cpu().numpy()    #[n,1]

        outputs_T = np.greater_equal(predictions, T)
        tp += np.sum(np.logical_and(outputs_T, targets), axis=0)
        tn += np.sum(np.logical_and(np.logical_not(outputs_T), np.logical_not(targets)), axis=0)
        fp += np.sum(np.logical_and(outputs_T, np.logical_not(targets)), axis=0)
        fn += np.sum(np.logical_and(np.logical_not(outputs_T), targets), axis=0)
        prec = tp / (tp + fp).astype(float)  # precision
        recall = tp / (tp + fn).astype(float)  # recall
        sens = tp / (tp + fn).astype(float)  # senstivity
        spec = tn / (tn + fp).astype(float)  # spec
        TPR = tp / (tp + fn).astype(float)
        FPR = fp / (tn + fp).astype(float)
        prec[np.isnan(prec)] = 0
        F1 = 2 * ((prec * sens) / (prec + sens))
        F1 = torch.tensor(np.nanmax(F1), device=device)   # F1 Score
        Recall = torch.tensor(recall, device=device)
        PR1 = torch.tensor(auc(recall, prec), device=device)
        PR = torch.tensor(np.trapz(y=recall, x=prec), device=device)  # average precision-recall value
        AUC = torch.tensor(np.trapz(y=sens, x=spec), device=device)


    return {"F1":F1, "PR1":PR1, "AUC":AUC, "PR":PR, "Recall":Recall, "PREC":prec,}
    # return {"F1":F1, "PR1":PR1,}

@coerce_numpy
def TS_compute_precisions(
    validation_step_outputs: torch.Tensor,
    minsep: int = 0,
    step: int = 0.001,
):
    T = np.arange(0, 1 + step, step)[None, :]  # threshold value [1,1001]
    tp = 0; fn = 0; fp = 0; tn = 0

    for batch in validation_step_outputs:
        predictions = batch["predictions"]
        targets = batch["tgt"]
        missing_nt_index = batch["missing_nt_index"]

        if predictions.dim() == 3:
            predictions = predictions.squeeze()     # [BS, L, L]-> [L,L]
        if targets.dim() == 3:
            targets = targets.squeeze()     # [BS, L, L]-> [L,L]

        seqlen, _ = predictions.size()
        targets_len, _ = targets.size()
        device = predictions.device

        mask_target = torch.triu(torch.ones((targets_len, targets_len), device=device), minsep+1) > 0    # 返回上三角矩阵，对角线偏移量1
        mask_pred = torch.triu(torch.ones((seqlen, seqlen), device=device), minsep+1) > 0
        for i in missing_nt_index:
            mask_pred[i, :] = 0
            mask_pred[:, i] = 0
        targets = targets.masked_select(mask_target)
        predictions = predictions.masked_select(mask_pred)

        targets = targets.unsqueeze(1).cpu().numpy()    #[n,1]
        predictions = predictions.unsqueeze(1).cpu().numpy()    #[n,1]

        outputs_T = np.greater_equal(predictions, T)
        tp += np.sum(np.logical_and(outputs_T, targets), axis=0)
        tn += np.sum(np.logical_and(np.logical_not(outputs_T), np.logical_not(targets)), axis=0)
        fp += np.sum(np.logical_and(outputs_T, np.logical_not(targets)), axis=0)
        fn += np.sum(np.logical_and(np.logical_not(outputs_T), targets), axis=0)
        prec = tp / (tp + fp).astype(float)  # precision
        recall = tp / (tp + fn).astype(float)  # recall
        sens = tp / (tp + fn).astype(float)  # senstivity
        spec = tn / (tn + fp).astype(float)  # spec
        TPR = tp / (tp + fn).astype(float)
        FPR = fp / (tn + fp).astype(float)
        prec[np.isnan(prec)] = 0
        F1 = 2 * ((prec * sens) / (prec + sens))
        F1 = torch.tensor(np.nanmax(F1), device=device)   # F1 Score
        Recall = torch.tensor(recall, device=device)
        PR1 = torch.tensor(auc(recall, prec), device=device)
        PR = torch.tensor(np.trapz(y=recall, x=prec), device=device)  # average precision-recall value
        AUC = torch.tensor(np.trapz(y=sens, x=spec), device=device)


    return {"F1":F1, "PR1":PR1, "AUC":AUC, "PR":PR, "Recall":Recall, "PREC":prec,}

@coerce_numpy
def TS_compute_precisions_premask(
    validation_step_outputs: torch.Tensor,
    minsep: int = 0,
    step: int = 0.001,
):
    T = np.arange(0, 1 + step, step)[None, :]  # threshold value [1,1001]
    tp = 0; fn = 0; fp = 0; tn = 0

    for batch in validation_step_outputs:
        predictions = batch["predictions"]
        targets = batch["tgt"]

        if predictions.dim() == 3:
            predictions = predictions.squeeze()     # [BS, L, L]-> [L,L]
        if targets.dim() == 3:
            targets = targets.squeeze()     # [BS, L, L]-> [L,L]

        seqlen, _ = predictions.size()
        device = predictions.device

        mask = torch.triu(torch.ones((seqlen, seqlen), device=device), minsep + 1) > 0  # 返回上三角矩阵，对角线偏移量1
        targets = targets.masked_select(mask)
        predictions = predictions.masked_select(mask)

        targets = targets.unsqueeze(1).cpu().numpy()  # [n,1]
        predictions = predictions.unsqueeze(1).cpu().numpy()  # [n,1]


        outputs_T = np.greater_equal(predictions, T)
        tp += np.sum(np.logical_and(outputs_T, targets), axis=0)
        tn += np.sum(np.logical_and(np.logical_not(outputs_T), np.logical_not(targets)), axis=0)
        fp += np.sum(np.logical_and(outputs_T, np.logical_not(targets)), axis=0)
        fn += np.sum(np.logical_and(np.logical_not(outputs_T), targets), axis=0)
        prec = tp / (tp + fp).astype(float)  # precision
        recall = tp / (tp + fn).astype(float)  # recall
        sens = tp / (tp + fn).astype(float)  # senstivity
        spec = tn / (tn + fp).astype(float)  # spec
        TPR = tp / (tp + fn).astype(float)
        FPR = fp / (tn + fp).astype(float)
        prec[np.isnan(prec)] = 0
        F1 = 2 * ((prec * sens) / (prec + sens))
        F1 = torch.tensor(np.nanmax(F1), device=device)   # F1 Score
        Recall = torch.tensor(recall, device=device)
        PR1 = torch.tensor(auc(recall, prec), device=device)
        PR = torch.tensor(np.trapz(y=recall, x=prec), device=device)  # average precision-recall value
        AUC = torch.tensor(np.trapz(y=sens, x=spec), device=device)

    return {"F1": F1, "PR1": PR1, "AUC": AUC, "PR": PR, "Recall": Recall, "PREC": prec, }

@coerce_numpy
def compute_precisions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    src_lengths: Optional[torch.Tensor] = None,
    minsep: int = 6,
    maxsep: Optional[int] = None,
    override_length: Optional[int] = None,  # for casp
):
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
    if targets.dim() == 2:
        targets = targets.unsqueeze(0)

    # Check sizes
    if predictions.size() != targets.size():
        raise ValueError(
            f"Size mismatch. Received predictions of size {predictions.size()}, "
            f"targets of size {targets.size()}"
        )
    device = predictions.device

    batch_size, seqlen, _ = predictions.size()
    seqlen_range = torch.arange(seqlen, device=device)

    sep = seqlen_range.unsqueeze(0) - seqlen_range.unsqueeze(1)
    sep = sep.unsqueeze(0)
    valid_mask = sep >= minsep
    valid_mask = valid_mask & (targets >= 0)  # negative targets are invalid

    if maxsep is not None:
        valid_mask &= sep < maxsep

    if src_lengths is not None:
        valid = seqlen_range.unsqueeze(0) < src_lengths.unsqueeze(1)
        valid_mask &= valid.unsqueeze(1) & valid.unsqueeze(2)
    else:
        src_lengths = torch.full([batch_size], seqlen, device=device, dtype=torch.long)

    predictions = predictions.masked_fill(~valid_mask, float("-inf"))

    x_ind, y_ind = np.triu_indices(seqlen, minsep)
    predictions_upper = predictions[:, x_ind, y_ind]
    targets_upper = targets[:, x_ind, y_ind]

    topk = seqlen if override_length is None else max(seqlen, override_length)
    indices = predictions_upper.argsort(dim=-1, descending=True)[:, :topk]
    topk_targets = targets_upper[torch.arange(batch_size).unsqueeze(1), indices]
    if topk_targets.size(1) < topk:
        topk_targets = F.pad(topk_targets, [0, topk - topk_targets.size(1)])

    cumulative_dist = topk_targets.type_as(predictions).cumsum(-1)

    gather_lengths = src_lengths.unsqueeze(1)
    if override_length is not None:
        gather_lengths = override_length * torch.ones_like(
            gather_lengths, device=device
        )

    gather_indices = (
        torch.arange(0.1, 1.1, 0.1, device=device).unsqueeze(0) * gather_lengths
    ).type(torch.long) - 1

    binned_cumulative_dist = cumulative_dist.gather(1, gather_indices)
    binned_precisions = binned_cumulative_dist / (gather_indices + 1).type_as(
        binned_cumulative_dist
    )

    pl5 = binned_precisions[:, 1]
    pl2 = binned_precisions[:, 4]
    pl = binned_precisions[:, 9]
    auc = binned_precisions.mean(-1)

    return {"AUC": auc, "P@L": pl, "P@L2": pl2, "P@L5": pl5}
