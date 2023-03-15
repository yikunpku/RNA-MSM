from typing import List, Tuple, Union, Iterator, Sequence, TextIO
from copy import copy
import contextlib
import math
import tempfile
import re
from pathlib import Path
import subprocess
import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist
from Bio import SeqIO
from Bio.Seq import Seq
from utils.typed import PathLike
current_directory = Path(__file__).parent.absolute()


class MSA:
    """Class that represents a multiple sequence alignment."""

    def __init__(
        self,
        sequences: List[Tuple[str, str]],
        seqid_cutoff: float = 0.2,
    ):
        self.headers = [header for header, _ in sequences]
        self.sequences = [seq for _, seq in sequences]
        self._seqlen = len(self.sequences[0])
        assert all(
            len(seq) == self._seqlen for seq in self.sequences
        ), "Seqlen Mismatch!"

        self._depth = len(self.sequences)
        self.seqid_cutoff = seqid_cutoff
        self.is_nucleotide = all(
            re.match(r"(A|C|T|G|U|-)*", seq) for seq in self.sequences
        )

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        return zip(self.headers, self.sequences)

    def select(self, indices: Sequence[int], axis: str = "seqs") -> "MSA":
        assert axis in ("seqs", "positions")
        if axis == "seqs":
            data = [(self.headers[idx], self.sequences[idx]) for idx in indices]
            return self.__class__(data)
        else:
            data = [
                (header, "".join(seq[idx] for idx in indices)) for header, seq in self
            ]
            return self.__class__(data)

    def swap(self, index1: int, index2: int) -> "MSA":
        headers = copy(self.headers)
        sequences = copy(self.sequences)
        headers[index1], headers[index2] = headers[index2], headers[index1]
        sequences[index1], sequences[index2] = sequences[index2], sequences[index1]
        data = list(zip(headers, sequences))
        return self.__class__(data, seqid_cutoff=self.seqid_cutoff)

    def filter_coverage(self, threshold: float, axis: str = "seqs") -> "MSA":
        assert 0 <= threshold <= 1
        assert axis in ("seqs", "positions")
        notgap = self.array != self.gap
        match = notgap.mean(1 if axis == "seqs" else 0)
        indices = np.where(match >= threshold)[0]
        return self.select(indices, axis=axis)

    def hhfilter(
        self,
        seqid: int = 90,
        diff: int = 0,
        cov: int = 0,
        qid: int = 0,
        qsc: float = -20.0,
        binary: str = "hhfilter",
    ) -> "MSA":

        with tempfile.TemporaryDirectory(dir=current_directory) as tempdirname:    # dir="/userhome/zhouyq/gaozhq/zhyk/projects/emb-transformer/data"
            tempdir = Path(tempdirname)
            fasta_file = tempdir / "input.fasta"
            fasta_file.write_text(
                "\n".join(f">{i}\n{seq}" for i, seq in enumerate(self.sequences))
            )
            output_file = tempdir / "output.fasta"
            command = " ".join(
                [
                    f"{binary}",
                    f"-i {fasta_file}",
                    "-M a2m",
                    f"-o {output_file}",
                    f"-id {seqid}",
                    f"-diff {diff}",
                    f"-cov {cov}",
                    f"-qid {qid}",
                    f"-qsc {qsc}",
                ]
            ).split(" ")
            result = subprocess.run(command, capture_output=True)
            result.check_returncode()
            with output_file.open() as f:
                indices = [int(line[1:].strip()) for line in f if line.startswith(">")]
            return self.select(indices, axis="seqs")

    def replace_(self, inp: str, rep: str) -> "MSA":
        dtype = self.dtype
        self.dtype = np.dtype("S1")  # type: ignore
        self.array[self.array == inp.encode()] = rep.encode()
        self.dtype = dtype
        return self

    @property
    def gap(self) -> Union[bytes, int]:
        return b"-" if self.dtype == np.dtype("S1") else ord("-")

    def __repr__(self) -> str:
        return f"MSA, L: {self.seqlen}, N: {self.depth}\n" f"{self.array}"

    def __getitem__(self, idx):
        return self.array[idx]

    def pdist(self) -> np.ndarray:
        dtype = self.dtype
        self.dtype = np.uint8
        dist = squareform(pdist(self.array, "hamming"))
        self.dtype = dtype
        return dist

    def greedy_select(self, num_seqs: int, mode: str = "max") -> "MSA":
        assert mode in ("max", "min")
        if self.depth <= num_seqs:
            return self
        dtype = self.dtype
        self.dtype = np.uint8

        optfunc = np.argmax if mode == "max" else np.argmin
        all_indices = np.arange(self.depth)
        indices = [0]
        pairwise_distances = np.zeros((0, self.depth))
        for _ in range(num_seqs - 1):
            dist = cdist(self.array[indices[-1:]], self.array, "hamming")
            pairwise_distances = np.concatenate([pairwise_distances, dist])
            shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
            shifted_index = optfunc(shifted_distance)
            index = np.delete(all_indices, indices)[shifted_index]
            indices.append(index)
        indices = sorted(indices)
        self.dtype = dtype
        return self.select(indices, axis="seqs")

    def sample_weights(self, num_seqs: int) -> "MSA":
        if self.depth <= num_seqs:
            return self
        weights = self.weights[1:]
        weights = weights / weights.sum()
        indices = (
            np.random.choice(
                self.depth - 1, size=num_seqs - 1, replace=False, p=weights
            )
            + 1
        )
        indices = np.sort(indices)
        indices = np.append(0, indices)
        return self.select(indices, axis="seqs")

    def select_diverse(self, num_seqs: int, method: str = "hhfilter") -> "MSA":
        assert method in ("hhfilter", "sample-pretrained", "diversity-max", "diversity-min")
        if num_seqs >= self.depth:
            return self

        if method == "hhfilter":
            msa = self.hhfilter(diff=num_seqs)  # diff=num_seqs
            if num_seqs < msa.depth:
                msa = msa.select(np.arange(num_seqs))
        elif method == "sample-pretrained":
            msa = self.sample_weights(num_seqs)
        elif method == "diversity-max":
            msa = self.greedy_select(num_seqs, mode="max")
        elif method == "diversity-min":
            msa = self.greedy_select(num_seqs, mode="min")

        return msa

    def invcov(self) -> np.ndarray:
        """given one-hot encoded MSA, return contacts"""
        from sklearn.preprocessing import OneHotEncoder
        dtype = self.dtype
        self.dtype = np.uint8
        Y = OneHotEncoder(drop=[self.gap]).fit_transform(self.array.reshape(-1, 1)).toarray().reshape(self.depth, self.seqlen, -1)
        K = Y.shape[-1]
        Y_flat = Y.reshape(self.depth, -1)
        c = np.cov(Y_flat.T)
        self.dtype = dtype
        return np.linalg.norm(c.reshape(self.seqlen, K, self.seqlen, K), ord=2, axis=(1, 3))
        # shrink = 4.5 / math.sqrt(self.depth) * np.eye(c.shape[0])
        # ic = np.linalg.inv(c + shrink)
        # ic = ic.reshape(self.seqlen, K, self.seqlen, K)
        # return apc(np.sqrt(np.square(ic).sum((1, 3))))

    @property
    def array(self) -> np.ndarray:
        if not hasattr(self, "_array"):
            self._array = np.array([list(seq) for seq in self.sequences], dtype="|S1")
        return self._array

    @property
    def dtype(self) -> type:
        return self.array.dtype

    @dtype.setter
    def dtype(self, value: type) -> None:
        assert value in (np.uint8, np.dtype("S1"))
        self._array = self.array.view(value)

    @property
    def seqlen(self) -> int:
        return self._seqlen

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def seqid_cutoff(self) -> float:
        return self._seqid_cutoff

    @seqid_cutoff.setter
    def seqid_cutoff(self, value: float) -> None:
        assert 0 <= value <= 1
        if getattr(self, "_seqid_cutoff", None) != value:
            with contextlib.suppress(AttributeError):
                delattr(self, "_weights")
            with contextlib.suppress(AttributeError):
                delattr(self, "_neff")
        self._seqid_cutoff = value

    @property
    def is_covered(self) -> np.ndarray:
        if not hasattr(self, "_is_covered"):
            self._is_covered = (self[1:] != self.gap).any(0)
        return self._is_covered

    @property
    def coverage(self) -> float:    # 比较每一列的‘-’数量，‘-’越多，coverage值越小
        if not hasattr(self, "_coverage"):
            notgap = self.array != self.gap
            self._coverage = notgap.mean(0)     # .mean(0):计算每一列的平均值
        return self._coverage

    @property
    def weights(self) -> np.ndarray:
        if not hasattr(self, "_weights"):
            self._weights = 1 / (self.pdist() < self.seqid_cutoff).sum(1)
        return self._weights

    @property
    def is_protein(self) -> bool:
        return not self.is_nucleotide

    def neff(self, normalization: Union[float, str] = "none") -> float:
        if isinstance(normalization, str):
            assert normalization in ("none", "sqrt", "seqlen")
            normalization = {
                "none": 1,
                "sqrt": math.sqrt(self.seqlen),
                "seqlen": self.seqlen,
            }[normalization]
        if not hasattr(self, "_neff"):
            self._neff = self.weights.sum()
        return self._neff / normalization

    @classmethod
    def from_stockholm(
        cls,
        stofile: Union[PathLike, TextIO],
        keep_insertions: bool = False,
        **kwargs,
    ) -> "MSA":

        output = []
        valid_indices = None
        for record in SeqIO.parse(stofile, "stockholm"):
            description = record.description
            sequence = str(record.seq)
            if not keep_insertions:
                if valid_indices is None:
                    valid_indices = [i for i, aa in enumerate(sequence) if aa != "-"]
                sequence = "".join(sequence[idx] for idx in valid_indices)
            output.append((description, sequence))
        return cls(output, **kwargs)

    @classmethod
    def from_fasta(
        cls,
        fasfile: Union[PathLike, TextIO],
        keep_insertions: bool = False,
        uppercase: bool = False,
        remove_lowercase_cols: bool = False,
        **kwargs,
    ) -> "MSA":

        output = []
        valid_indices = None
        for record in SeqIO.parse(fasfile, "fasta"):
            description = record.description
            sequence = str(record.seq)
            if remove_lowercase_cols:
                if valid_indices is None:
                    valid_indices = [i for i, aa in enumerate(sequence) if aa.isupper()]
                sequence = "".join(sequence[i] for i in valid_indices)
            if not keep_insertions:
                sequence = re.sub(r"([a-z]|\.|\*)", "", sequence)
                sequence = re.sub(r"[T]", "U", sequence)
                sequence = re.sub(r"[RYKMSWBDHVN]", "X", sequence)
            if uppercase:
                sequence = sequence.upper()
            output.append((description, sequence))
        return cls(output, **kwargs)

    @classmethod
    def from_file(
        cls,
        alnfile: PathLike,
        keep_insertions: bool = False,
        **kwargs,
    ) -> "MSA":
        filename = Path(alnfile)
        if filename.suffix == ".sto":
            return cls.from_stockholm(filename, keep_insertions, **kwargs)
        elif filename.suffix in (".fas", ".fasta", ".a3m", ".a2m"):
            return cls.from_fasta(filename, keep_insertions, **kwargs)
        else:
            raise ValueError(f"Unknown file format {filename.suffix}")

    @classmethod
    def from_sequences(cls, sequences: Sequence[str]) -> "MSA":
        return cls([("", seq) for seq in sequences])

    def write(self, outfile: PathLike, form: str = "fasta") -> None:
        SeqIO.write(
            (SeqIO.SeqRecord(Seq(seq), id=header, description="") for header, seq in self),
            outfile,
            form,
        )
