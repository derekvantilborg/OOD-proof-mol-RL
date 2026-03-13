"""SMILES tokenization and encoding utilities."""

import re
import numpy as np
from typing import Optional
from molrl.chem_constants import VOCAB


# Compile the tokenizer regex once at module load time.
_ATOM_TOKENS = ['Cl', 'Br', 'H', 'C', 'c', 'N', 'n', 'O', 'o', 'F', 'P', 'p', 'S', 's', 'I']
_PATTERN = (
    r"(\[|\]|" + "|".join(_ATOM_TOKENS) + r"|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\d)"
)
_REGEX = re.compile(_PATTERN)


class TokenizationError(ValueError):
    """Raised when a SMILES string cannot be encoded with the current vocab."""


def smiles_tokenizer(smiles: str) -> list[str]:
    """Tokenize a SMILES string into a list of token strings."""
    return _REGEX.findall(smiles)


def smiles_to_encoding(smi: str) -> list[int]:
    """Encode a single SMILES string to a fixed-length integer vector.

    Raises:
        TokenizationError: if any character in the SMILES is not recognised by
                           the tokenizer (i.e. the joined tokens don't fully
                           reconstruct the input), or if the sequence (including
                           start/end tokens) exceeds VOCAB['max_len'].
    """
    tokens = smiles_tokenizer(smi)

    # If the regex didn't capture every character, there are unknown tokens.
    if "".join(tokens) != smi:
        raise TokenizationError(f"SMILES contains characters not in vocab: {smi!r}")

    # +2 for start and end tokens
    if len(tokens) + 2 > VOCAB['max_len']:
        raise TokenizationError(
            f"SMILES too long ({len(tokens) + 2} tokens > max {VOCAB['max_len']}): {smi!r}"
        )

    encoding = (
        [VOCAB['start_idx']]
        + [VOCAB['token_indices'][t] for t in tokens]
        + [VOCAB['end_idx']]
    )
    encoding += [VOCAB['pad_idx']] * (VOCAB['max_len'] - len(encoding))
    return encoding


def encode_smiles_list(
    smiles_list: list[str],
    errors: str = "drop",
) -> tuple[np.ndarray, list[str], list[tuple[int, str]]]:
    """Encode a list of SMILES strings into a 2-D int16 array for HDF5 storage.

    Args:
        smiles_list: SMILES to encode.
        errors:  "drop"  — silently skip invalid SMILES (default).
                 "raise" — re-raise the first TokenizationError encountered.

    Returns:
        encodings:  np.ndarray of shape (N_valid, max_len), dtype int16.
        valid_smiles: list of SMILES that were successfully encoded (same order).
        failed:     list of (original_index, error_message) for every skipped entry.
    """
    if errors not in ("drop", "raise"):
        raise ValueError(f"errors must be 'drop' or 'raise', got {errors!r}")

    encodings = []
    valid_smiles = []
    failed = []

    for i, smi in enumerate(smiles_list):
        try:
            encodings.append(smiles_to_encoding(smi))
            valid_smiles.append(smi)
        except TokenizationError as e:
            if errors == "raise":
                raise
            failed.append((i, str(e)))

    arr = np.array(encodings, dtype=np.int16) if encodings else np.empty((0, VOCAB['max_len']), dtype=np.int16)
    return arr, valid_smiles, failed


def is_tokenizable(smi: str) -> bool:
    """Return True if the SMILES can be fully encoded with the current vocab."""
    try:
        smiles_to_encoding(smi)
        return True
    except TokenizationError:
        return False


def encoding_to_smiles(encoding: list[int] | np.ndarray) -> Optional[str]:
    """Decode a token-index vector back to a SMILES string.

    Strips padding, start, and end tokens. Returns None if the sequence
    contains no tokens between start and end.
    """
    indices_token = VOCAB['indices_token']
    special = {VOCAB['pad_idx'], VOCAB['start_idx'], VOCAB['end_idx']}

    tokens = [indices_token[idx] for idx in encoding if idx not in special and idx in indices_token]
    return "".join(tokens) if tokens else None


if __name__ == "__main__":

    # Example usage
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    encoding = smiles_to_encoding(smiles)
    print("Encoding:", encoding)
    decoded = encoding_to_smiles(encoding)
    print("Decoded SMILES:", decoded)