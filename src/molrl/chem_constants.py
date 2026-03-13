"""Chemical constants for data cleaning and analysis."""

from __future__ import annotations


# Vocabulary for SMILES encoding
VOCAB = {
    'pad_char': '_',
    'start_char': '>',
    'end_char': ';',
    'max_len': 102,
    'vocab_size': 36,
    'pad_idx': 0,
    'start_idx': 1,
    'end_idx': 35,
    'indices_token': {
        0: '_', 1: '>', 2: 'C', 3: 'c', 4: '(', 5: ')', 6: 'O', 7: '1', 8: '=', 9: 'N', 10: '2',
        11: '@', 12: '[', 13: ']', 14: 'H', 15: 'n', 16: '3', 17: 'F', 18: '4', 19: 'S', 20: '/',
        21: 'Cl', 22: 's', 23: '5', 24: 'o', 25: '#', 26: '\\', 27: 'Br', 28: 'P', 29: '6', 30: 'I',
        31: '7', 32: '8', 33: 'p', 34: '-', 35: ';'
    },
    'token_indices': {
        '_': 0, '>': 1, 'C': 2, 'c': 3, '(': 4, ')': 5, 'O': 6, '1': 7, '=': 8, 'N': 9, '2': 10,
        '@': 11, '[': 12, ']': 13, 'H': 14, 'n': 15, '3': 16, 'F': 17, '4': 18, 'S': 19, '/': 20,
        'Cl': 21, 's': 22, '5': 23, 'o': 24, '#': 25, '\\': 26, 'Br': 27, 'P': 28, '6': 29, 'I': 30,
        '7': 31, '8': 32, 'p': 33, '-': 34, ';': 35
    }
}


COMMON_SOLVENTS = [
    'O',
    'O=[N+]([O-])O',
    'F[P-](F)(F)(F)(F)F',
    'O=C([O-])C(F)(F)F',
    'O=C(O)CC(O)(CC(=O)O)C(=O)O',
    'CCO',
    'CCN(CC)CC',
    '[O-][Cl+3]([O-])([O-])O',
    'O=P(O)(O)O',
    'O=C(O)/C=C/C(=O)O',
    'O=C(O)/C=C\\C(=O)O',
    '[O-][Cl+3]([O-])([O-])[O-]',
    'CS(=O)(=O)O',
    'O=C(O)C(=O)O',
    'F[B-](F)(F)F',
    'C',
    'Cc1ccc(S(=O)(=O)[O-])cc1',
    'C1CCC(NC2CCCCC2)CC1',
    'O=CO',
    'O=S(=O)([O-])O',
    'O=C(O)C(F)(F)F',
    'COS(=O)(=O)[O-]',
    'CN(C)C=O',
    'Cc1ccc(S(=O)(=O)O)cc1',
    'O=C(O)CCC(=O)O',
    'O=C(O)[C@H](O)[C@@H](O)C(=O)O',
    'CS(=O)(=O)[O-]',
    'c1ccncc1',
    'NCCO',
    'O=S(=O)([O-])C(F)(F)F',
    'CNC[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO',
    'O=C(O)C(O)C(O)C(=O)O',
    'CC(=O)O',
    'NC(CO)(CO)CO',
    'O=S(=O)(O)O'
    ]

SMARTS_NEUTRALIZATION_PATTERNS = (
    # Imidazoles
    ('[n+;H]', 'n'),
    # Amines
    ('[N+;!H0]', 'N'),
    # Carboxylic acids and alcohols
    ('[$([O-]);!$([O-][#7])]', 'O'),
    # Thiols
    ('[S-;X1]', 'S'),
    # Sulfonamides
    ('[$([N-;X2]S(=O)=O)]', 'N'),
    # Enamines
    ('[$([N-;X2][C,N]=C)]', 'N'),
    # Tetrazoles
    ('[n-]', '[nH]'),
    # Sulfoxides
    ('[$([S-]=O)]', 'S'),
    # Amides
    ('[$([N-]C=O)]', 'N'),
)

SMARTS_COMMON_SALTS = "[Cl,Na,Mg,Ca,K,Br,Zn,Ag,Al,Li,I,O,N,H]"
