from utils.dataset import (
    BaseWrapperDataset,
)
from typing import Collection
from utils.dataset import CollatableVocabDataset
from utils.align import MSA
from typing import Union
from Bio import SeqIO
from utils.typed import PathLike
from torch.utils.data import DataLoader
from typing import Optional
from utils.tokenization import Vocab
from pathlib import Path
import torch
import numpy as np


class A2MDataset(torch.utils.data.Dataset):
    """
    Creates a dataset from a directory of a2m files.
    Args:
        data_file (Union[str, Path]): Path to directory of a2m files.
        split_files (Optional[Collection[str]]): Subset of files to use,
            can be used to specify training / validation / testing sets.
        max_seqs_per_msa (Optional[Collection[int]]): Maximum number of
            rna msa sequences.
        sample_method (str): Method to sample rna sequences from msa file,
            when rna sequence entries in the msa file is greater than the
            max_seqs_per_msa.
    """

    def __init__(
            self,
            data_file: PathLike,
            split_files: Optional[Collection[str]] = None,
            max_seqs_per_msa: Optional[int] = None,
            sample_method: str = "hhfilter",
    ):
        assert sample_method in ("hhfilter", "sample-pretrained", "diversity-max", "diversity-min")
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        if not data_file.is_dir():
            raise NotADirectoryError(data_file)

        file_glob = data_file.glob("*.a2m_msa2")
        if split_files is None:
            file_list = list(file_glob)
        else:
            split_files = set(split_files)
            if len(split_files) == 0:
                raise ValueError("Passed an empty split file set")

            file_list = [f for f in file_glob if f.stem.split('.')[0] in split_files]
            if len(file_list) != len(split_files):
                num_missing = len(split_files) - len(file_list)
                raise FileNotFoundError(
                    f"{num_missing} specified split files not found in directory"
                )

        if len(file_list) == 0:
            raise FileNotFoundError(f"No .a2m_msa2 files found in {data_file}")

        self._file_list = sorted(file_list)
        self.keys = {f.stem: i for i, f in enumerate(self._file_list)}
        self._max_seqs_per_msa = max_seqs_per_msa
        self._sample_method = sample_method

    def get(self, key: str):
        index = self.keys[key]
        return self[index]

    def key(self, index: int) -> str:
        return self._file_list[index].stem

    def __len__(self) -> int:
        return len(self._file_list)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._max_seqs_per_msa == 1:
            seq = str(next(SeqIO.parse(self._file_list[index], "fasta")).seq)
            return seq
        else:
            msa = MSA.from_fasta(self._file_list[index])
            if self._max_seqs_per_msa is not None:
                msa = msa.select_diverse(
                    self._max_seqs_per_msa, method=self._sample_method
                )
            return msa


class RNADataset(CollatableVocabDataset):
    def __init__(
            self,
            data_path: PathLike,
            msa_path: PathLike,
            vocab: Vocab,
            split_files: Optional[Collection[str]] = None,
            max_seqs_per_msa: Optional[int] = 64,
            sample_method: str = "hhfilter",
    ):
        super().__init__(vocab)

        data_path = Path(data_path)
        msa_path = Path(msa_path)

        self.rna_id = split_files
        self.a3m_data = A2MDataset(
            data_path / msa_path,
            split_files=split_files,
            max_seqs_per_msa=max_seqs_per_msa,
            sample_method=sample_method,
        )

    def __len__(self) -> int:
        return len(self.a3m_data)

    def __getitem__(self, index):
        rna_id = self.rna_id[index]
        msa = self.a3m_data[index]
        tokens = torch.from_numpy(self.vocab.encode(msa))

        return rna_id, tokens


class RandomCropDataset(BaseWrapperDataset):
    def __init__(
            self,
            dataset: CollatableVocabDataset,
            max_seqlen: int
    ):
        super().__init__(dataset)

        self.sizes = max_seqlen
        self.max_seqlen = max_seqlen
        self.num_special = int(self.vocab.prepend_bos) + int(self.vocab.append_eos)
        self.max_seqlen_no_special = self.max_seqlen - self.num_special
        self.sizes = np.minimum(self.sizes, max_seqlen)  # type: ignore

    def __getitem__(self, idx):
        rna_id, tokens = self.dataset[idx]
        seqlen = tokens.size(-1)

        if seqlen > self.max_seqlen:
            low_idx = int(self.vocab.prepend_bos)
            high_idx = seqlen - int(self.vocab.append_eos)
            start_idx = np.random.randint(low_idx, seqlen - self.max_seqlen)
            end_idx = start_idx + self.max_seqlen_no_special
            tokens = torch.cat(
                [
                    tokens[..., :low_idx],
                    tokens[..., start_idx:end_idx],
                    tokens[..., high_idx:],
                ],
                -1,
            )
        return rna_id, tokens