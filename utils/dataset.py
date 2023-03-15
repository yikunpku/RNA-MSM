from typing import (
    List,
    Any,
    Optional,
    Collection,
    TypeVar,
    Sequence,
    Union,
    Tuple,
    Callable,
    Dict,
)
import threading
import math
from pathlib import Path
import torch
import torch.distributed as dist
import numpy as np
from operator import methodcaller
import subprocess
import json
import os
from Bio import SeqIO
import numba
from utils.tensor import collate_tensors, numpy_seed
from utils.typed import PathLike
from utils.align import MSA
from utils.tokenization import Vocab


T = TypeVar("T")


class ThreadsafeFile:
    def __init__(
        self,
        filepath: PathLike,
        open_func: Callable[[PathLike], T],
        close_func: Callable[[T], None] = methodcaller("close"),
    ):
        self._threadlocal = threading.local()
        self._filepath = filepath
        self._open_func = open_func
        self._close_func = close_func

    def __getattr__(self, name: str):
        return getattr(self.file, name)

    @property
    def file(self) -> T:
        if not hasattr(self._threadlocal, "file"):
            self._threadlocal.file = self._open_func(self._filepath)
        return self._threadlocal.file

    def __getstate__(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "_threadlocal"}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__ = state
        self._threadlocal = threading.local()

    def __del__(self):
        if hasattr(self._threadlocal, "file"):
            self._close_func(self._threadlocal.file)
            del self._threadlocal.file


class SizedDataset(torch.utils.data.Dataset):
    def __init__(self, sizes: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)  # type: ignore
        self._sizes = sizes

    def __len__(self):
        return len(self.sizes)

    @property
    def sizes(self):
        return self._sizes


class CollatableDataset(torch.utils.data.Dataset):
    def collater(self, batch: List[Any]) -> Any:
        try:
            return torch.stack(batch, 0)
        except Exception:
            return batch


class CollatableVocabDataset(CollatableDataset):
    def __init__(self, vocab: Vocab, *args, **kwargs):
        super().__init__(*args, **kwargs)  # type: ignore
        self.vocab = vocab

    def collater(self, batch: List[Any]) -> Any:
        return collate_tensors(batch, constant_value=self.vocab.pad_idx)


class BaseWrapperDataset(CollatableVocabDataset):
    """BaseWrapperDataset. Wraps an existing dataset.

    Args:
        dataset (torch.utils.data.dataset): Dataset to wrap.
    """

    def __init__(self, dataset: CollatableVocabDataset):
        super().__init__(dataset.vocab)
        self.dataset = dataset

    def __getattr__(self, name: str):
        if "dataset" not in self.__dict__:
            raise AttributeError("No dataset")
        return getattr(self.dataset, name)

    def __getitem__(self, index: int):
        return self.dataset[index]

    def collater(self, batch):
        return self.dataset.collater(batch)

    def __len__(self):
        return len(self.dataset)


class SubsetDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: CollatableVocabDataset,
        subset: Sequence[float],
        index: int,
        seed: int = 0,
    ):
        super().__init__(dataset)
        fracs = np.array(subset)
        assert np.isclose(fracs.sum(), 1)
        percentages = np.append(0, np.cumsum(fracs))
        percentages[-1] = 1
        with numpy_seed(seed):
            indices = np.random.permutation(np.arange(len(dataset)))  # type: ignore
            start, end = (
                percentages[index : index + 2] * len(dataset)  # type: ignore
            ).astype(np.int64)
            indices = np.sort(indices[start:end])
        self._indices = indices
        self.sizes = dataset.sizes[indices]  # type: ignore

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index: int):
        index = self._indices[index]
        return super().__getitem__(index)


class JsonDataset(torch.utils.data.Dataset):
    """Creates a dataset from a directory of json files.
    Args:
        data_file (Union[str, Path]): Path to directory of lm_npz files
        split_files (Optional[Collection[str]]): Subset of files to use,
            can be used to specify training / validation / testing sets.
        json_file Optional[Collection[str]]: Json file which contains
            labels for split_files.
    """

    def __init__(
        self,
        data_path: PathLike,
        split_files: Optional[Collection[str]] = None,
        json_file: Optional[Collection[str]] = None,
    ):
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(data_path)
        if not data_path.is_dir():
            raise NotADirectoryError(data_path)

        split_files = list(split_files)
        if len(split_files) == 0:
            raise ValueError("Passed an empty split file set")
        json_file = open(os.path.join(data_path, json_file), "rb")
        self._json_file = json_file.read()
        self._split_files = split_files

    def __len__(self) -> int:
        return len(self._split_files)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)
        item = json.loads(self._json_file)[self._split_files[index]]
        return item

class NPZDataset(torch.utils.data.Dataset):
    """Creates a dataset from a directory of lm_npz files.
    Args:
        data_file (Union[str, Path]): Path to directory of lm_npz files
        split_files (Optional[Collection[str]]): Subset of files to use,
            can be used to specify training / validation / testing sets.
    """

    def __init__(
        self,
        data_file: PathLike,
        split_files: Optional[Collection[str]] = None,
        lazy: bool = False,
    ):
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        if not data_file.is_dir():
            raise NotADirectoryError(data_file)

        file_glob = data_file.glob("*.npz")
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
            raise FileNotFoundError(f"No .npz files found in {data_file}")

        self._file_list = sorted(file_list)
        self.keys = {f.stem: i for i, f in enumerate(self._file_list)}
        self._lazy = lazy

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

        item = np.load(self._file_list[index])
        if not self._lazy:
            item = dict(item)
        return item


class A3MDataset(torch.utils.data.Dataset):
    """Creates a dataset from a directory of a3m files.
    Args:
        data_file (Union[str, Path]): Path to directory of a3m files
        split_files (Optional[Collection[str]]): Subset of files to use,
            can be used to specify training / validation / testing sets.
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

        file_glob = data_file.glob("*.a2m") # *.a2m
        if split_files is None:
            file_list = list(file_glob)
        else:
            split_files = set(split_files)
            if len(split_files) == 0:
                raise ValueError("Passed an empty split file set")

            file_list = [f for f in file_glob if f.stem in split_files]
            if len(file_list) != len(split_files):
                num_missing = len(split_files) - len(file_list)
                raise FileNotFoundError(
                    f"{num_missing} specified split files not found in directory"
                )

        if len(file_list) == 0:
            raise FileNotFoundError(f"No .a2m files found in {data_file}")

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


class EncodedA3MDataset(CollatableVocabDataset, A3MDataset):
    def __init__(
        self,
        data_file: PathLike,
        vocab: Vocab,
        split_files: Optional[Collection[str]] = None,
        max_seqs_per_msa: Optional[int] = None,
        sample_method: str = "hhfilter",
    ):
        super().__init__(
            data_file=data_file,
            vocab=vocab,
            split_files=split_files,
            max_seqs_per_msa=max_seqs_per_msa,
            sample_method=sample_method,
        )

    def __getitem__(self, idx):
        return self.vocab.encode(super().__getitem__(idx))


class FastaDataset(SizedDataset):
    """
    For loading protein sequence datasets in the common FASTA data format

    Modified from github.com/pytorch/fairseq.
    """

    def __init__(self, data_file: PathLike, cache_indices: bool = False):
        self.data_file = Path(data_file)
        if not self.data_file.exists():
            raise FileNotFoundError(
                f"{self.data_file}\n"
                "If using hydra, make sure you are using abolute instead of relative paths."
            )
        self.file = ThreadsafeFile(data_file, open)
        self.cache = Path(f"{data_file}.idx.npy")
        if cache_indices:
            if self.cache.exists():
                self.offsets, sizes = np.load(self.cache)
            else:
                self.offsets, sizes = self._build_index()
                np.save(self.cache, np.stack([self.offsets, sizes]))
        else:
            self.offsets, sizes = self._build_index()

        super().__init__(sizes)

    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, idx: int):
        self.file.seek(self.offsets[idx])
        if idx == len(self) - 1:
            data = self.file.read()
        else:
            data = self.file.read(self.offsets[idx + 1] - self.offsets[idx])
        desc, *seq = data.split("\n")
        return desc[1:], "".join(seq)

    def __len__(self):
        return self.offsets.size

    def _build_index(self):
        # Use grep and awk to get 100M/s on local SSD.
        # Should process your enormous 100G fasta in ~10 min single core...
        bytes_offsets = subprocess.check_output(
            f"cat {self.data_file} | tqdm --bytes --total $(wc -c < {self.data_file})"
            "| grep --byte-offset '^>' -o | cut -d: -f1",
            shell=True,
        )
        fasta_lengths = subprocess.check_output(
            f"cat {self.data_file} | tqdm --bytes --total $(wc -c < {self.data_file})"
            '| awk \'/^>/ {print "";next;} { printf("%s",$0);}\' | tail -n+2 | awk '
            "'{print length($1)}'",
            shell=True,
        )
        bytes_np = np.fromstring(bytes_offsets, dtype=np.int64, sep=" ")
        sizes_np = np.fromstring(fasta_lengths, dtype=np.int64, sep=" ")
        return bytes_np, sizes_np


class EncodedFastaDataset(CollatableVocabDataset, FastaDataset):
    def __init__(self, data_file: PathLike, vocab: Vocab):
        super().__init__(data_file=data_file, vocab=vocab, cache_indices=True)
        self._sizes += int(self.vocab.prepend_bos) + int(self.vocab.append_eos)

    def __getitem__(self, index: int) -> torch.Tensor:
        desc, seq = super().__getitem__(index)
        return torch.from_numpy(self.vocab.encode_single_sequence(seq))


class TorchDataset(CollatableVocabDataset):
    def __init__(self, data_file: PathLike, vocab: Vocab):
        data_file = Path(data_file)
        self.data_file = data_file
        self.data = torch.load(data_file)
        self.offsets, self.sizes = np.load(data_file.with_suffix(".fasta.idx.npy"))

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        item = self.data[self.offsets[idx] : self.offsets[idx] + self.sizes[idx]]
        return item

    def collater(self, batch):
        return collate_tensors(batch, constant_value=self.vocab.pad_idx)


class MaxTokenBatch(object):
    def __init__(self, max_tokens: int, pad_idx: int):
        self.max_tokens = max_tokens
        self.pad_idx = pad_idx
        self.items: List[torch.Tensor] = []
        self.sizes = None

    def can_add_item(self, item: torch.Tensor) -> bool:
        sizes = np.asarray(item.size())
        if self.sizes is not None:
            sizes = np.max([self.sizes, sizes], 0)
        total_tokens = (len(self.items) + 1) * sizes.prod()
        return total_tokens <= self.max_tokens

    def add_item(self, item: torch.Tensor):
        self.items.append(item)
        sizes = np.asarray(item.size())
        if self.sizes is None:
            self.sizes = sizes
        else:
            self.sizes = np.max([self.sizes, sizes], 0)
        if self.num_tokens > self.max_tokens:
            raise RuntimeError("Too many sequences in batch!")

    def finalize(self) -> torch.Tensor:
        return collate_tensors(self.items, constant_value=self.pad_idx)

    @property
    def num_tokens(self) -> int:
        if self.sizes is None:
            return 0
        else:
            return len(self.items) * self.sizes.prod()


BatchOrSequence = TypeVar("BatchOrSequence", MaxTokenBatch, Sequence[MaxTokenBatch])


class AutoBatchingDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, dataset: CollatableVocabDataset, max_tokens: int, shuffle: bool = False
    ):
        super().__init__()
        self.dataset = dataset
        self.vocab = dataset.vocab
        self.max_tokens = max_tokens
        self.shuffle = shuffle

    def maybe_make_and_add_batch(
        self,
        batch: Optional[BatchOrSequence],
        item: Union[torch.Tensor, Sequence[torch.Tensor]],
    ) -> Tuple[BatchOrSequence, bool]:
        if batch is None:
            if isinstance(item, torch.Tensor):
                batch = MaxTokenBatch(  # type: ignore
                    self.max_tokens, self.vocab.pad_idx
                )
            else:
                batch = [  # type: ignore
                    MaxTokenBatch(self.max_tokens, self.vocab.pad_idx) for _ in item
                ]

        if isinstance(batch, MaxTokenBatch):
            can_add = batch.can_add_item(item)  # type: ignore
            if can_add:
                batch.add_item(item)  # type: ignore
        else:
            can_add = batch[0].can_add_item(item[0])  # type: ignore
            if can_add:
                for b, i in zip(batch, item):  # type: ignore
                    b.add_item(i)
        return batch, can_add  # type: ignore

    def __iter__(self):
        indices = np.arange(len(self.dataset))

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            worker_rank = dist.get_rank()
        else:
            world_size = 1
            worker_rank = 0

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            world_size *= worker_info.num_workers
            worker_rank = worker_rank * worker_rank.num_workers + worker_info.id

        chunk_size = math.ceil(len(indices) / world_size)
        indices = indices[chunk_size * worker_rank : chunk_size * (worker_rank + 1)]

        if self.shuffle:
            indices = np.random.permutation(indices)

        batch = None
        for idx in indices:
            items = self.dataset[idx]
            batch, added = self.maybe_make_and_add_batch(batch, items)
            if not added:
                if isinstance(batch, MaxTokenBatch):
                    yield batch.finalize()
                else:
                    yield type(items)(b.finalize() for b in batch)
                batch, added = self.maybe_make_and_add_batch(None, items)
                assert added, "Item size too large to include!"
        if batch:
            if isinstance(batch, MaxTokenBatch):
                yield batch.finalize()
            else:
                yield type(items)(b.finalize() for b in batch)


@numba.njit
def batch_by_size(
    indices: np.ndarray, sizes: np.ndarray, max_tokens: int
) -> List[List[int]]:
    batches: List[List[int]] = []
    batch: List[int] = [0][:0]
    batch_size = 0
    for i in range(len(indices)):
        idx = indices[i]
        size = sizes[i]
        if size > max_tokens:
            raise RuntimeError("An item was too large to batch.")
        if size + batch_size > max_tokens:
            batches.append(batch)
            batch = [0][:0]
            batch_size = 0
        batch.append(idx)
        batch_size += size
    batches.append(batch)
    return batches


class BatchBySequenceLength(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset: SizedDataset,
        max_tokens: int,
        shuffle=True,
        seed=0,
    ):
        super().__init__(dataset)
        indices = np.argsort(dataset.sizes)
        sizes = dataset.sizes[indices]
        batches = batch_by_size(indices, sizes, max_tokens)

        self.dataset = dataset
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.batches = batches
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.batches), generator=g).tolist()
        else:
            indices = list(range(len(self.batches)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == len(self)
        yield from (self.batches[idx] for idx in indices)

    def __len__(self):
        return math.ceil(len(self.batches) / self.num_replicas)

    @property
    def num_replicas(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1

    @property
    def total_size(self) -> int:
        return len(self) * self.num_replicas

    @property
    def rank(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        else:
            return 0

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all
        replicas use a different random ordering for each epoch. Otherwise, the next
        iteration of this sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class RandomCropDataset(BaseWrapperDataset):
    def __init__(self, dataset: CollatableVocabDataset, max_seqlen: int):
        super().__init__(dataset)
        # add
        self.sizes = max_seqlen
        #
        self.max_seqlen = max_seqlen
        self.num_special = int(self.vocab.prepend_bos) + int(self.vocab.append_eos)
        self.max_seqlen_no_special = self.max_seqlen - self.num_special
        self.sizes = np.minimum(self.sizes, max_seqlen)  # type: ignore

    def __getitem__(self, idx):
        item = self.dataset[idx]
        seqlen = item.size(-1)
        if seqlen > self.max_seqlen:
            low_idx = int(self.vocab.prepend_bos)
            high_idx = seqlen - int(self.vocab.append_eos)
            start_idx = np.random.randint(low_idx, high_idx)
            end_idx = start_idx + self.max_seqlen_no_special
            item = torch.cat(
                [
                    item[..., :low_idx],
                    item[..., start_idx:end_idx],
                    item[..., high_idx:],
                ],
                -1,
            )
        return item


class SubsampleMSADataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: CollatableVocabDataset,
        max_tokens: int,
        max_seqs: Optional[int] = None,
    ):
        super().__init__(dataset)
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs if max_seqs is not None else float("inf")

    def __getitem__(self, idx):
        msa = self.dataset[idx]

        num_alignments, seqlen = msa.size()
        max_alignments = self.max_tokens // seqlen
        max_alignments = min(self.max_seqs, max_alignments)
        if max_alignments < num_alignments:
            indices = np.random.randint(1, num_alignments, size=max_alignments - 1)
            indices = np.append(0, indices)
            msa = msa[indices]

        return msa


class MaskedTokenWrapperDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: CollatableVocabDataset,
        mask_prob: float = 0.15,
        random_token_prob: float = 0.1,
        leave_unmasked_prob: float = 0.1,
    ):
        # TODO - add column masking?
        # TODO - add collater
        super().__init__(dataset)
        assert 0 <= mask_prob <= 1
        assert 0 <= random_token_prob <= 1
        assert 0 <= leave_unmasked_prob <= 1

        self._mask_prob = mask_prob
        self._random_token_prob = random_token_prob
        self._leave_unmasked_prob = leave_unmasked_prob

    def __getitem__(self, idx):
        item = self.dataset[idx]
        random_probs = torch.rand_like(item, dtype=torch.float)
        random_probs[(item == self.vocab.bos_idx) | (item == self.vocab.eos_idx)] = 1   # <cls>和<eos>的特殊符号不会被mask
        do_mask = random_probs < self.mask_prob # 小于0.15为True

        tgt = item.masked_fill(~do_mask, self.vocab.pad_idx)    # 生成label，没有被mask的为1，mask的显示为真实的idx
        mask_with_token = random_probs < (
            self.mask_prob * (1 - self.leave_unmasked_prob) # 替换为mask token 小于0.15*0.9为True
        )
        src = item.masked_fill(mask_with_token, self.vocab.mask_idx)    #用mask_idx(11)去遮盖
        mask_with_random = random_probs < (self.mask_prob * self.random_token_prob) # 随机替换为其他token 小于0.15*0.1为True
        # TODO - maybe prevent special tokens?
        rand_tokens = torch.randint_like(src, len(self.vocab))
        src[mask_with_random] = rand_tokens[mask_with_random]
        return src, tgt

    @property
    def mask_prob(self) -> float:
        return self._mask_prob

    @property
    def random_token_prob(self) -> float:
        return self._random_token_prob

    @property
    def leave_unmasked_prob(self) -> float:
        return self._leave_unmasked_prob

    def collater(self, batch: List[Any]) -> Any:
        src = collate_tensors(
            [el[0] for el in batch],
            constant_value=self.vocab.pad_idx,
        )
        tgt = collate_tensors(
            [el[1] for el in batch],
            constant_value=self.vocab.pad_idx,
        )
        return src, tgt
