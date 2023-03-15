from typing import List, Tuple
import re
from itertools import product, zip_longest
import pandas as pd
import numpy as np

_FASTA_VOCAB = "ARNDCQEGHILKMFPSTWYV"


def single_mutant_names(sequence: str) -> List[str]:
    """Returns the names of all single mutants of a sequence."""
    mutants = []
    for (i, wt), mut in product(enumerate(sequence), _FASTA_VOCAB):
        if wt == mut:
            continue
        mutant = f"{wt}{i + 1}{mut}"
        mutants.append(mutant)
    return mutants


def split_mutant_name(mutant: str) -> Tuple[str, int, str]:
    """Splits a mutant name into the wildtype, position, and mutant."""
    return mutant[0], int(mutant[1:-1]), mutant[-1]


def make_mutation(sequence: str, mutant: str, start_ind: int = 1) -> str:
    """Makes a mutation on a particular sequence. Multiple mutations may be separated
    by ',', ':', or '+', characters.
    """
    delimiters = [",", r"\+", ":"]
    expression = re.compile("|".join(delimiters))
    if mutant.upper() == "WT":
        return sequence
    if expression.search(mutant):
        mutants = expression.split(mutant)
        for mutant in mutants:
            sequence = make_mutation(sequence, mutant)
        return sequence
    else:
        wt, pos, mut = split_mutant_name(mutant)
        assert sequence[pos - start_ind] == wt
        return sequence[: pos - start_ind] + mut + sequence[pos - start_ind + 1 :]


def create_mutant_df(sequence: str) -> pd.DataFrame:
    """Create a dataframe with mutant names and sequences"""
    names = ["WT"] + single_mutant_names(sequence)
    sequences = [sequence] + [make_mutation(sequence, mut) for mut in names[1:]]
    return pd.DataFrame({"mutant": names, "sequence": sequences})


def seqdiff(seq1: str, seq2: str) -> str:
    diff = []
    for aa1, aa2 in zip_longest(seq1, seq2, fillvalue="-"):
        if aa1 == aa2:
            diff.append(" ")
        else:
            diff.append("|")
    out = f"{seq1}\n{''.join(diff)}\n{seq2}"
    return out


def to_pivoted_mutant_df(df: pd.DataFrame) -> pd.DataFrame:
    df["wt_aa"] = df["mutant"].str.get(0)
    df["mut_aa"] = df["mutant"].str.get(-1)
    df["Position"] = df["mutant"].str.slice(1, -1).astype(int)
    df = df.drop(columns="mutant").pivot(
        index="mut_aa", columns=["Position", "wt_aa"]
    )
    df = df.loc[list(_FASTA_VOCAB)]
    return df


def pivoted_mutant_df(sequence: str, scores: np.ndarray) -> pd.DataFrame:
    index = pd.Index(list(_FASTA_VOCAB), name="mut_aa")
    columns = pd.MultiIndex.from_arrays(
        [list(range(1, len(sequence) + 1)), list(sequence)], names=["Position", "wt_aa"]
    )
    df = pd.DataFrame(
        data=scores,
        index=index,
        columns=columns,
    )
    return df
