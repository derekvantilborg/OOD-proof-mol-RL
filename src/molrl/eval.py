# Metrics (R², RMSE, validity, uniqueness, novelty), chemical quality
# (QED, SA, PAINS), reward hacking detection.

from rdkit.Chem import MolFromSmiles
from rdkit import RDLogger


def smiles_validity(smiles_list):
    # RDKit emits parser warnings for invalid SMILES; suppress during batch scoring.
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")

    def is_valid_smiles(smi: str) -> int:
        try:
            mol = MolFromSmiles(smi)
            if mol and mol.GetNumAtoms() > 0:
                return 1
            else:
                return 0
        except Exception:
            return 0

    try:
        valid_count = sum([is_valid_smiles(smi) for smi in smiles_list])
        return valid_count / len(smiles_list) if smiles_list else 0.0
    finally:
        RDLogger.EnableLog("rdApp.warning")
        RDLogger.EnableLog("rdApp.error")


def smiles_novelty(generated_smiles, training_smiles):
    training_set = set(training_smiles)
    novel_count = sum(1 for smi in generated_smiles if smi not in training_set)
    return novel_count / len(generated_smiles) if generated_smiles else 0.0


def smiles_uniqueness(smiles_list):
    unique_smiles = set(smiles_list)
    return len(unique_smiles) / len(smiles_list) if smiles_list else 0.0


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]
