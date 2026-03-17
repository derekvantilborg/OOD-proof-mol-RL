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