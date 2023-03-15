from typing import Optional, NamedTuple, Dict
import numpy as np
from .typed import PathLike
from .constants import IUPAC_CODES
from scipy.spatial.distance import squareform, pdist
import contextlib
from collections import defaultdict


class PDB_SPEC(object):
    __slots__ = ()
    ID = slice(0, 6)
    RESIDUE = slice(17, 20)
    RESN = slice(22, 27)
    ATOM = slice(12, 16)
    CHAIN = slice(21, 22)
    X = slice(30, 38)
    Y = slice(38, 46)
    Z = slice(46, 54)


PDB_SPEC = PDB_SPEC()  # type: ignore


class NotAnAtomLine(Exception):
    """Raised if input line is not an atom line"""

    pass


class AtomLine(NamedTuple):

    ID: str
    RESIDUE: str
    RESN: str
    ATOM: str
    CHAIN: str
    X: float
    Y: float
    Z: float

    @classmethod
    def from_line(cls, line: str) -> "AtomLine":
        id_ = line[PDB_SPEC.ID].strip()

        if id_ == "HETATM" and line[PDB_SPEC.RESIDUE] == "MSE":
            line = line.replace("HETATM", "ATOM  ")
            line = line.replace("MSE", "MET")
        elif not line[PDB_SPEC.ID].startswith("ATOM"):
            raise NotAnAtomLine(line)

        residue = line[PDB_SPEC.RESIDUE]
        resn = line[PDB_SPEC.RESN].strip()
        atom = line[PDB_SPEC.ATOM].strip()
        chain = line[PDB_SPEC.CHAIN]
        x = float(line[PDB_SPEC.X])
        y = float(line[PDB_SPEC.Y])
        z = float(line[PDB_SPEC.Z])

        return cls(id_, residue, resn, atom, chain, x, y, z)


class Structure(object):
    def __init__(self, sequence: str, coords: np.ndarray, residues: np.array):
        assert len(sequence) == coords.shape[0]
        assert coords.ndim == 3
        assert residues.ndim == 1

        if coords.shape[1] == 3:
            cbeta = self.extend_cbeta(coords)
            coords = np.concatenate([coords, cbeta[:, None]], 1)

        self._sequence = sequence
        self._coords = coords
        self._residues = residues

    @property
    def coords(self) -> np.ndarray:
        return self._coords

    @property
    def sequence(self) -> str:
        return self._sequence

    @property
    def residues(self) -> np.array:
        return self._residues

    @property
    def distogram(self) -> np.array:
        if not hasattr(self, "_distogram"):
            self._distogram = squareform(pdist(self.coords[:, -1]))
        return self._distogram

    @property
    def contacts(self) -> np.array:
        return self.distogram < 8

    def __len__(self) -> int:
        return len(self.sequence)

    @staticmethod
    def iterate_atomlines(path: PathLike):
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                with contextlib.suppress(NotAnAtomLine):
                    yield AtomLine.from_line(line)

    @staticmethod
    def _extend(a, b, c, L, A, D):
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

    @staticmethod
    def extend_cbeta(coords):
        """ Return inferred position of Cb atom from positions of C, N, CA atoms
        """
        assert coords.ndim == 3
        assert coords.shape[1:] == (3, 3)
        N = coords[:, 0]
        CA = coords[:, 1]
        C = coords[:, 2]

        Cbeta = Structure._extend(C, N, CA, 1.522, 1.927, -2.143)
        return Cbeta

    @classmethod
    def from_pdb(
        cls,
        path: PathLike,
        chain: Optional[str] = None,
    ) -> "Structure":
        """
        input:  x = PDB filename
                atoms = atoms to extract (optional)
        output: (length, atoms, coords=(x,y,z)), sequence
        """

        def chain_valid(line: AtomLine):
            return chain is None or line.CHAIN == chain

        def resn_alnum(line: AtomLine):
            return line.RESN.isdecimal()

        sequence: Dict[int, str] = {}
        residues: Dict[int, Dict[str, np.ndarray]] = defaultdict(
            lambda: defaultdict(list)
        )
        for line in filter(
            resn_alnum, filter(chain_valid, cls.iterate_atomlines(path))
        ):
            resn = int(line.RESN)
            residues[resn][line.ATOM] = np.array([line.X, line.Y, line.Z])
            sequence[resn] = line.RESIDUE

        minres = min(residues.keys())
        maxres = max(residues.keys())
        seqlen = maxres - minres + 1

        coords = np.zeros([seqlen, 3, 3])
        seq_string = "".join(
            IUPAC_CODES.get(sequence.get(i, "X").capitalize(), "X")
            for i in range(minres, maxres + 1)
        )

        for resn in range(minres, maxres + 1):
            coord_resn = coords[resn - minres]
            if resn in residues:
                for i, atom in enumerate(["N", "CA", "C"]):
                    coord_resn[i] = residues[resn].get(atom, np.full(3, np.nan))
            else:
                coord_resn[:] = np.full_like(coord_resn, np.nan)

        valid_resn = np.array(sorted(residues.keys()))

        return Structure(seq_string, coords, valid_resn - 1)
