from typing import Tuple, List, Dict, Optional, Sequence
from collections import defaultdict
import string
from pathlib import Path
from Bio import SeqIO
import subprocess
from .typed import PathLike
from .constants import IUPAC_CODES
from .dataset import ThreadsafeFile
import numpy as np
from scipy.spatial.distance import squareform, pdist
import pandas as pd


def read_sequences(
    filename: PathLike,
    remove_insertions: bool = False,
    remove_gaps: bool = False,
) -> Tuple[List[str], List[str]]:

    filename = Path(filename)
    if filename.suffix == ".sto":
        form = "stockholm"
    elif filename.suffix in (".fas", ".fasta", ".a3m"):
        form = "fasta"
    else:
        raise ValueError(f"Unknown file format {filename.suffix}")

    translate_dict: Dict[str, Optional[str]] = {}
    if remove_insertions:
        translate_dict.update(dict.fromkeys(string.ascii_lowercase))
    else:
        translate_dict.update(dict(zip(string.ascii_lowercase, string.ascii_uppercase)))

    if remove_gaps:
        translate_dict["-"] = None

    translate_dict["."] = None
    translate_dict["*"] = None
    translation = str.maketrans(translate_dict)

    def process_record(record: SeqIO.SeqRecord) -> Tuple[str, str]:
        description = record.description
        sequence = str(record.seq).translate(translation)
        return description, sequence

    headers = []
    sequences = []
    for header, seq in map(process_record, SeqIO.parse(str(filename), form)):
        headers.append(header)
        sequences.append(seq)
    return headers, sequences


def read_first_sequence(
    filename: PathLike,
    remove_insertions: bool = False,
    remove_gaps: bool = False,
) -> Tuple[str, str]:

    filename = Path(filename)
    if filename.suffix == ".sto":
        form = "stockholm"
    elif filename.suffix in (".fas", ".fasta", ".a3m"):
        form = "fasta"
    else:
        raise ValueError(f"Unknown file format {filename.suffix}")

    translate_dict: Dict[str, Optional[str]] = {}
    if remove_insertions:
        translate_dict.update(dict.fromkeys(string.ascii_lowercase))
    else:
        translate_dict.update(dict(zip(string.ascii_lowercase, string.ascii_uppercase)))

    if remove_gaps:
        translate_dict["-"] = None

    translate_dict["."] = None
    translate_dict["*"] = None
    translation = str.maketrans(translate_dict)

    def process_record(record: SeqIO.SeqRecord) -> Tuple[str, str]:
        description = record.description
        sequence = str(record.seq).translate(translation)
        return description, sequence

    return process_record(next(SeqIO.parse(str(filename), form)))


def count_sequences(seqfile: PathLike) -> int:
    "Utility for quickly counting sequences in a fasta/a3m file."
    num_seqs = subprocess.check_output(f'grep "^>" -c {seqfile}', shell=True)
    return int(num_seqs)


def parse_PDB(x, atoms=["N", "CA", "C"], chain=None):
    """
    input:  x = PDB filename
            atoms = atoms to extract (optional)
    output: (length, atoms, coords=(x,y,z)), sequence
    """
    xyz, seq, min_resn, max_resn = {}, {}, np.inf, -np.inf
    for line in open(x, "rb"):
        line = line.decode("utf-8", "ignore").rstrip()

        if line[:6] == "HETATM" and line[17 : 17 + 3] == "MSE":
            line = line.replace("HETATM", "ATOM  ")
            line = line.replace("MSE", "MET")

        if line[:4] == "ATOM":
            ch = line[21:22]
            if ch == chain or chain is None:
                atom = line[12 : 12 + 4].strip()
                resi = line[17 : 17 + 3]
                resn = line[22 : 22 + 5].strip()
                x, y, z = [float(line[i : (i + 8)]) for i in [30, 38, 46]]

                if resn[-1].isalpha():
                    resa, resn = resn[-1], int(resn[:-1]) - 1
                else:
                    resa, resn = "", int(resn) - 1
                if resn < min_resn:
                    min_resn = resn
                if resn > max_resn:
                    max_resn = resn
                if resn not in xyz:
                    xyz[resn] = {}
                if resa not in xyz[resn]:
                    xyz[resn][resa] = {}
                if resn not in seq:
                    seq[resn] = {}
                if resa not in seq[resn]:
                    seq[resn][resa] = resi

                if atom not in xyz[resn][resa]:
                    xyz[resn][resa][atom] = np.array([x, y, z])

    # convert to numpy arrays, fill in missing values
    seq_, xyz_ = [], []
    for resn in range(min_resn, max_resn + 1):
        if resn in seq:
            for k in sorted(seq[resn]):
                seq_.append(IUPAC_CODES.get(seq[resn][k].capitalize(), "X"))
        else:
            seq_.append("X")
        if resn in xyz:
            for k in sorted(xyz[resn]):
                for atom in atoms:
                    if atom in xyz[resn][k]:
                        xyz_.append(xyz[resn][k][atom])
                    else:
                        xyz_.append(np.full(3, np.nan))
        else:
            for atom in atoms:
                xyz_.append(np.full(3, np.nan))

    valid_resn = np.array(sorted(xyz.keys()))
    return np.array(xyz_).reshape(-1, len(atoms), 3), "".join(seq_), valid_resn


def extend(a, b, c, L, A, D):
    """
    input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
    output: 4th coord
    """

    def normalize(x):
        return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True)

    bc = normalize(b - c)
    n = normalize(np.cross(b - a, bc))
    m = [bc, np.cross(n, bc), n]
    d = [L * np.cos(A), L * np.sin(A) * np.cos(D), -L * np.sin(A) * np.sin(D)]
    return c + sum([m * d for m, d in zip(m, d)])


def contacts_from_pdb(
    filename: PathLike, distance_threshold: float = 8.0
) -> np.ndarray:
    coords, _, _ = parse_PDB(filename)

    N = coords[:, 0]
    CA = coords[:, 1]
    C = coords[:, 2]

    Cbeta = extend(C, N, CA, 1.522, 1.927, -2.143)
    distogram = squareform(pdist(Cbeta))
    return distogram < distance_threshold


class UniProtView(Sequence[Dict[str, str]]):
    def __init__(self, path: PathLike):
        self.path = Path(path)
        self.cache = self.path.with_name(self.path.name + ".idx.npy")
        self.file = ThreadsafeFile(self.path, open)
        if self.cache.exists():
            self.offsets = np.load(self.cache)
        else:
            self.offsets = self._build_index()
            self._num_sequences = len(self.offsets)
            np.save(self.cache, self.offsets)

    def finalize(self, item: Dict[str, List[str]], join: str = ""):
        deletewhite = str.maketrans(dict.fromkeys(string.whitespace))
        output = {key: join.join(values) for key, values in item.items()}
        output["sequence"] = (
            output["SQ"].split("\n", maxsplit=1)[1].translate(deletewhite)
        )
        return output

    def __iter__(self):
        entry: Dict[str, List[str]] = defaultdict(list)

        with open(self.path) as f:
            for line in f:
                if line.startswith("ID"):
                    if entry:
                        yield self.finalize(entry)
                    entry = defaultdict(list)
                data = line[5:]
                if line[:5].strip():
                    tag = line[:5].strip()
                entry[tag].append(data)
            yield self.finalize(entry)

    def count_sequences(self):
        return int(
            subprocess.run(
                ["grep", "-c", "^ID", str(self.path)], capture_output=True
            ).stdout.decode()
        )

    def _build_index(self):
        # Use grep and awk to get 100M/s on local SSD.
        # Should process your enormous 100G fasta in ~10 min single core...
        bytes_offsets = subprocess.check_output(
            f"cat {self.path} | tqdm --bytes --total $(wc -c < {self.path})"
            "| grep --byte-offset '^ID' -o | cut -d: -f1",
            shell=True,
        )
        bytes_np = np.fromstring(bytes_offsets, dtype=np.int64, sep=" ")
        return bytes_np

    def __getitem__(self, idx):
        self.file.seek(self.offsets[idx])
        if idx == len(self) - 1:
            data = self.file.read()
        else:
            data = self.file.read(self.offsets[idx + 1] - self.offsets[idx])

        entry: Dict[str, List[str]] = defaultdict(list)
        for line in data.split("\n"):
            data = line[5:]
            if line[:5].strip():
                tag = line[:5].strip()
            entry[tag].append(data)

        return self.finalize(entry, join="\n")

    def __len__(self):
        if not hasattr(self, "_num_sequences"):
            self._num_sequences = self.count_sequences()
        return self._num_sequences


def parse_uniprot(path: PathLike) -> Sequence[Dict[str, str]]:
    return UniProtView(path)


def parse_simple_pdb(path: PathLike) -> pd.DataFrame:
    names = [
        "record",
        "atomno",
        "atom",
        "resn",
        "chain",
        "resi",
        "x",
        "y",
        "z",
        "occupancy",
        "plddt",
        "element",
    ]
    df = pd.read_csv(path, sep=r"\s+", names=names)
    df = df[df["record"] == "ATOM"].reset_index().drop("index", axis="columns")
    df["atomno"] = df["atomno"].astype(int)
    df["resi"] = df["resi"].astype(int)
    return df
