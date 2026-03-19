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
