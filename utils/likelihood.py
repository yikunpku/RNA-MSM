from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import msm
from typing import Union, List, Sequence, Optional, Tuple
from functools import partial
import pandas as pd


from .tokenization import Vocab
from .tensor import batched_iterator
from .align import MSA


IntSeq = Union[Sequence[int], np.ndarray, torch.Tensor]


def mask_and_repeat(tokens: torch.Tensor, indices: IntSeq, vocab: Vocab) -> torch.Tensor:
    seqlen = len(indices)
    repeat_dims = [1] * tokens.dim()
    tokens = tokens.unsqueeze(0).repeat(seqlen, *repeat_dims)
    if tokens.dim() == 2:
        tokens[torch.arange(seqlen), indices] = vocab.mask_idx
    else:
        tokens[torch.arange(seqlen), 0, indices] = vocab.mask_idx
    return tokens


def select_indices(logits: torch.Tensor, batch_indices, position_indices):
    if logits.dim() == 3:
        out = logits[batch_indices, position_indices]
    else:
        out = logits[batch_indices, 0, position_indices]
    return out


@torch.no_grad()
def sequence_logits(
    model: msm.model.ProteinBertModel,
    vocab: Vocab,
    sequence: Union[str, MSA],
    verbose: bool = False,
    mask_positions: bool = True,
    max_tokens: int = 2 ** 14,
    indices: Optional[IntSeq] = None,
    parallel: bool = False,
) -> torch.Tensor:

    device = next(model.parameters()).device
    tokens = torch.from_numpy(vocab.encode(sequence)).to(device)
    if parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        max_tokens *= torch.cuda.device_count()

    if indices is None:
        start = int(vocab.prepend_bos)
        end = tokens.size(-1) - int(vocab.append_eos)
        indices = torch.arange(start, end)

    seqlen = len(indices)
    if mask_positions:
        tokens = mask_and_repeat(tokens, indices, vocab)

        logits = torch.zeros((seqlen, len(vocab)), device=device)
        batch_size = max(1, max_tokens // np.prod(tokens.size()[1:]))
        for i, batch in enumerate(
            batched_iterator(tokens, batch_size=batch_size, device=device, verbose=verbose)
        ):  # type: Tuple[int, torch.Tensor]
            idx = i * batch_size

            batch_indices = torch.arange(batch.size(0))
            position_indices = indices[idx : idx + batch_size]
            out = model(batch)["logits"]
            out = select_indices(out, batch_indices, position_indices)
            logits[idx : idx + batch_size] = out
    else:
        out = model(tokens.unsqueeze(0))["logits"]
        logits = select_indices(out, torch.arange(seqlen), indices)
    return logits


@torch.no_grad()
def sequence_mutant_scores(
    model: msm.model.ProteinBertModel,
    vocab: Vocab,
    sequence: Union[str, MSA],
    verbose: bool = False,
    mask_positions: bool = True,
    max_tokens: int = 2 ** 14,
    indices: Optional[IntSeq] = None,
    parallel: bool = False,
) -> pd.DataFrame:
    logits = sequence_logits(
        model,
        vocab,
        sequence,
        verbose,
        mask_positions,
        max_tokens,
        indices,
        parallel,
    ).log_softmax(-1)
    wt_encoded = torch.from_numpy(vocab.encode(sequence))
    if wt_encoded.dim() == 2:
        wt_encoded = wt_encoded[0]

    if indices is None:
        start = int(vocab.prepend_bos)
        end = wt_encoded.size(-1) - int(vocab.append_eos)
        indices = torch.arange(start, end)

    wt_encoded = wt_encoded[indices]  # type: ignore

    wt_logits = logits[torch.arange(len(wt_encoded)), wt_encoded].unsqueeze(1)
    scores = logits - wt_logits

    alphabet = Vocab.from_fasta_standard().tokens[:20]
    remap = [vocab.index(tok) for tok in alphabet]
    scores = scores[:, remap].cpu().numpy()

    index = pd.Index(list(alphabet), name="mut_aa")
    columns = pd.MultiIndex.from_arrays(
        [
            indices.tolist() if isinstance(indices, torch.Tensor) else indices,
            [vocab.token(idx) for idx in wt_encoded],
        ],
        names=["Position", "wt_aa"],
    )
    df = pd.DataFrame(
        data=scores.T,
        index=index,
        columns=columns,
    )
    return df


@torch.no_grad()
def sequence_pseudo_ppl(
    model: msm.model.ProteinBertModel,
    vocab: Vocab,
    sequence: str,
    mask_positions: bool = True,
    verbose: bool = False,
    max_tokens: int = 2 ** 14,
    reduction: str = "mean",
    log: bool = False,
) -> float:

    device = next(model.parameters()).device
    tokens = torch.from_numpy(vocab.encode(sequence)).to(device)
    start = int(vocab.prepend_bos)
    end = tokens.size(-1) - int(vocab.append_eos)
    residue_tokens = tokens[start:end]

    logits = sequence_logits(
        model,
        vocab,
        sequence,
        mask_positions=mask_positions,
        verbose=verbose,
        max_tokens=max_tokens,
    )

    pseudo_ppl = nn.CrossEntropyLoss(reduction=reduction)(
        logits.view(-1, len(vocab)), residue_tokens
    )
    if not log:
        pseudo_ppl = pseudo_ppl.exp()
    return pseudo_ppl.item()


@torch.no_grad()
def pseudo_ppl(
    model: msm.model.ProteinBertModel,
    alphabet_or_vocab: Union[msm.data.Alphabet, Vocab],
    sequences: List[str],
    mask_positions: bool = True,
    max_tokens: int = 2 ** 14,
    log: bool = False,
):
    if not isinstance(alphabet_or_vocab, Vocab):
        vocab = Vocab.from_esm_alphabet(alphabet_or_vocab)
    else:
        vocab = alphabet_or_vocab

    model = model.cuda().eval()

    compute = partial(
        sequence_pseudo_ppl,
        model,
        vocab,
        max_tokens=max_tokens,
        mask_positions=mask_positions,
        log=log,
    )

    pseudo_ppl = []
    for sequence in tqdm(sequences):
        pppl = compute(sequence)
        pseudo_ppl.append(pppl)
    return torch.tensor(pseudo_ppl, dtype=torch.float)
