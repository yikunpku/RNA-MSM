from typing import Dict, Optional, Union, Sequence
import torch
import tape
import transformers
import numpy as np
import msm
from copy import copy
import logging
from utils.align import MSA
import itertools

logger = logging.getLogger(__name__)


class Vocab(object):
    def __init__(
        self,
        tokens: Dict[str, int],
        bos_token: str = "<cls>",
        eos_token: str = "<sep>",
        unk_token: Optional[str] = None,
        pad_token: str = "<pad>",
        mask_token: Optional[str] = None,
        prepend_bos: bool = True,
        append_eos: bool = True,
    ):
        if prepend_bos and bos_token not in tokens:
            raise KeyError(f"bos token '{bos_token}' not in input tokens.")
        if append_eos and eos_token not in tokens:
            raise KeyError(f"eos token '{eos_token}' not in input tokens.")
        if unk_token is not None and unk_token not in tokens:
            raise KeyError(f"unk token '{unk_token}' not in input tokens.")
        if pad_token not in tokens:
            raise KeyError(f"pad token '{pad_token}' not in input tokens.")
        if mask_token is not None and mask_token not in tokens:
            raise KeyError(f"mask token '{mask_token}' not in input tokens.")

        # prevent modifications to original dictionary from having an effect.
        tokens = copy(tokens)
        for tok in list(tokens.keys()):
            if len(tok) > 1 and tok not in {
                bos_token,
                eos_token,
                unk_token,
                mask_token,
                pad_token,
            }:
                logger.warning(f"Vocab contains non-special token of length > 1: {tok}")

        self.prepend_bos = prepend_bos
        self.append_eos = append_eos

        if prepend_bos:
            self.bos_token: Optional[str] = bos_token
            self.bos_idx = tokens[bos_token]
        else:
            self.bos_token = None
            self.bos_idx = -1
        if append_eos:
            self.eos_token: Optional[str] = eos_token
            self.eos_idx = tokens[eos_token]
        else:
            self.eos_token = None
            self.eos_idx = -1

        self.tokens_to_idx = tokens
        self.tokens = list(tokens.keys())
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.pad_token = pad_token

        self.pad_idx = tokens[pad_token]

        self.allow_unknown = unk_token is not None
        if unk_token is not None:
            self.unk_idx = tokens[unk_token]

        self.allow_mask = mask_token is not None
        if mask_token is not None:
            self.mask_idx = tokens[mask_token]

        self.uint8_symbols = np.sort(
            np.array([tok for tok in self.tokens if len(tok) == 1], dtype="|S1").view(
                np.uint8
            )
        )
        self.numpy_indices = np.array(
            [self.index(chr(tok)) for tok in self.uint8_symbols],
            dtype=np.int64,
        )

    def index(self, token: str) -> int:
        return self.tokens_to_idx[token]

    def token(self, index: int) -> str:
        return self.tokens[index]

    def __len__(self):
        return len(self.tokens)

    def __repr__(self) -> str:
        return f"Vocab({self.to_dict()})"

    def to_dict(self) -> Dict[str, int]:
        return copy(self.tokens_to_idx)

    def _convert_uint8_array(self, array: np.ndarray) -> np.ndarray:
        assert array.dtype in (np.dtype("S1"), np.uint8)
        array = array.view(np.uint8)
        mask = ~np.isin(array, self.uint8_symbols)
        if mask.any():
            array = array.copy()
            array[mask] = ord(self.unk_token)  # type: ignore
        locs = np.digitize(array, self.uint8_symbols, right=True)
        indices = self.numpy_indices[locs.reshape(-1)].reshape(locs.shape)
        return indices

    def add_special_tokens(self, array: np.ndarray) -> np.ndarray:
        pad_widths = [(0, 0)] * (array.ndim - 1) + [
            (int(self.prepend_bos), int(self.append_eos))
        ]
        return np.pad(
            array,
            pad_widths,
            constant_values=[(self.bos_idx, self.eos_idx)],
        )

    def encode_array(self, array: np.ndarray) -> np.ndarray:
        return self.add_special_tokens(self._convert_uint8_array(array))

    def encode_single_sequence(self, sequence: str) -> np.ndarray:
        return self.encode_array(np.array(list(sequence), dtype="|S1"))

    def encode_batched_sequences(self, sequences: Sequence[str]) -> np.ndarray:
        batch_size = len(sequences)
        max_seqlen = max(len(seq) for seq in sequences)
        extra_token_pad = int(self.prepend_bos) + int(self.append_eos)
        indices = np.full(
            (batch_size, max_seqlen + extra_token_pad), fill_value=self.pad_idx
        )

        for i, seq in enumerate(sequences):
            encoded = self.encode_single_sequence(seq)
            indices[i, : len(encoded)] = encoded
        return indices

    def decode_single_sequence(self, array: np.ndarray) -> str:
        array = array[int(self.prepend_bos) : len(array) - int(self.append_eos)]
        return "".join(self.token(idx) for idx in array)

    def encode(
        self, inputs: Union[str, Sequence[str], np.ndarray, MSA], validate: bool = True
    ) -> np.ndarray:
        if validate and not self.check_valid(inputs):
            raise ValueError("Invalid tokens in input")
        if isinstance(inputs, str):
            return self.encode_single_sequence(inputs)
        elif isinstance(inputs, Sequence):
            return self.encode_batched_sequences(inputs)
        elif isinstance(inputs, np.ndarray):
            return self.encode_array(inputs)
        elif isinstance(inputs, MSA) or hasattr(inputs, "array"):
            return self.encode_array(inputs.array)
        else:
            raise TypeError(f"Unknown input type {type(inputs)}")

    def check_valid(self, inputs: Union[str, Sequence[str], np.ndarray, MSA]) -> bool:
        if isinstance(inputs, str):
            tokens = set(inputs)
        elif isinstance(inputs, Sequence):
            tokens = set(itertools.chain.from_iterable(inputs))
        elif isinstance(inputs, np.ndarray):
            inputs = inputs.astype(np.dtype("S1"))
            tokens = {x.decode() for x in inputs.flatten()}
        elif isinstance(inputs, MSA) or hasattr(inputs, "sequences"):
            tokens = set(itertools.chain.from_iterable(inputs.sequences))
        else:
            raise TypeError(f"Unknown input type {type(inputs)}")
        return not bool(tokens - set(self.tokens))

    def decode(
        self, tokens: Union[np.ndarray, torch.Tensor]
    ) -> Union[str, Sequence[str]]:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()

        if tokens.ndim == 1:
            return self.decode_single_sequence(tokens)
        elif tokens.ndim == 2:
            return [self.decode_single_sequence(toks) for toks in tokens]
        elif tokens.ndim == 3:
            assert tokens.shape[0] == 1, "Cannot decode w/ batch size > 1"
            tokens = tokens[0]
            return self.decode(tokens)
        else:
            raise ValueError("Too many dimensions!")

    @classmethod
    def from_esm_alphabet(cls, alphabet: msm.data.Alphabet) -> "Vocab":
        return cls(
            tokens=alphabet.tok_to_idx,
            bos_token="<cls>",
            eos_token="<eos>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            prepend_bos=alphabet.prepend_bos,
            append_eos=alphabet.append_eos,
        )

    @classmethod
    def from_tape_tokenizer(cls, tokenizer: tape.tokenizers.TAPETokenizer) -> "Vocab":
        if "<unk>" in tokenizer.vocab:
            unk_token: Optional[str] = "<unk>"
        elif "X" in tokenizer.vocab:
            unk_token = "X"
        else:
            unk_token = None

        return cls(
            tokens=tokenizer.vocab,
            bos_token=tokenizer.start_token,
            eos_token=tokenizer.stop_token,
            unk_token=unk_token,
            pad_token="<pad>",
            mask_token=tokenizer.mask_token,
            prepend_bos=True,
            append_eos=True,
        )

    @classmethod
    def from_huggingface_tokenizer(
        cls, tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase
    ) -> "Vocab":
        return cls(
            tokens=tokenizer.get_vocab(),
            bos_token=tokenizer.cls_token,
            eos_token=tokenizer.sep_token,
            unk_token=tokenizer.unk_token,
            pad_token=tokenizer.pad_token,
            mask_token=tokenizer.mask_token,
            prepend_bos=tokenizer.cls_token is not None,
            append_eos=tokenizer.sep_token is not None,
        )

    @classmethod
    def from_fasta_standard(cls) -> "Vocab":
        alphabet = "ARNDCQEGHILKMFPSTWYV-X"
        a2n = {a: n for n, a in enumerate(alphabet)}
        return cls(
            a2n, pad_token="-", prepend_bos=False, append_eos=False, unk_token="X"
        )

    @classmethod
    def from_trrosetta(cls) -> "Vocab":
        alphabet = "ARNDCQEGHILKMFPSTWYV-"
        a2n = {a: n for n, a in enumerate(alphabet)}
        return cls(
            a2n, pad_token="-", prepend_bos=False, append_eos=False, unk_token="-"
        )


def test_encode_sequence():
    sequence = "LFKLGAENIFLGRKAATKEEAIRFAGEQLVKGGYVEPEYVQAMLDREKLTPTYLGESIAVPHGTVEAK"
    alphabet = msm.data.Alphabet.from_architecture("ESM-1b")
    vocab = Vocab.from_esm_alphabet(alphabet)
    batch_converter = alphabet.get_batch_converter()
    _, _, esm_tokens = batch_converter([("", sequence)])
    evo_tokens = vocab.encode(sequence)[None]
    assert (esm_tokens == evo_tokens).all()
