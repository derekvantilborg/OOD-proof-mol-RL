# SMILES cleaning, canonicalisation, ECFP computation, Tanimoto similarity,
# scaffold splitting, medicinal chemistry filters.

import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
from typing import Tuple, List, Optional, Dict, Union
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MolToSmiles, AllChem, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold 
from rdkit.Chem.rdchem import Mol
from typing import Tuple, List, Optional
import pandas as pd
import warnings
from rdkit import RDLogger
from rdkit.Chem import (
    MolFromSmiles, 
    MolToSmiles, 
    MolFromSmarts, 
    RemoveStereochemistry
)
from rdkit.Chem.AllChem import ReplaceSubstructs
from rdkit.Chem.SaltRemover import SaltRemover
import jax.numpy as jnp
from jax import vmap

from molrl.chem_constants import COMMON_SOLVENTS, SMARTS_NEUTRALIZATION_PATTERNS, SMARTS_COMMON_SALTS

# Suppress RDKit's internal deprecation warnings for MolStandardize submodules
# We're using rdMolStandardize (the C++ module) which is the correct approach,
# but RDKit's __init__.py imports deprecated Python wrappers
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='rdkit')
    from rdkit.Chem.MolStandardize import rdMolStandardize


def tanimoto(a, b):
    dot = jnp.dot(a, b)
    return dot / (a.sum() + b.sum() - dot)

# do fps = jnp.array(fps, dtype=jnp.float32) first
tanimoto_one_to_many = vmap(tanimoto, (None, 0))

# do fps = jnp.array(fps, dtype=jnp.float32) first
tanimoto_pairwise = vmap(vmap(tanimoto, (None, 0)), (0, None))


def fetch_chembl_bioactivity(
    target_chembl_id: str,
    standard_types: list[str] = ("Ki", "EC50"),
    relations: list[str] = ("=",),
) -> pd.DataFrame:
    """Fetch raw bioactivity data for a ChEMBL target.

    Returns one row per assay entry with the SMILES, numeric value, units,
    assay description, and a handful of metadata columns useful for curation.
    Only entries whose standard_type and relation match the supplied filters
    are returned; no other filtering or cleaning is applied here.

    Args:
        target_chembl_id: ChEMBL target ID (e.g. "CHEMBL203" for EGFR).
        standard_types: Endpoint types to keep (default: Ki and EC50).
        relations: Relation symbols to keep (default: exact "=" only,
                   excludes ">" / "<" qualifiers).

    Returns:
        DataFrame with columns:
            smiles                    – canonical SMILES from ChEMBL
            standard_type             – Ki / EC50 / etc.
            value                     – numeric value in standard_units
            units                     – unit string (usually "nM")
            assay_description         – free-text assay description
            assay_chembl_id           – assay identifier
            molecule_chembl_id        – compound identifier
            document_chembl_id        – publication reference
            data_validity_comment     – non-null flags potential issues
    """
    standard_types = set(standard_types)
    relations = set(relations)

    rows = []
    activity = new_client.activity
    results = activity.filter(target_chembl_id=target_chembl_id)

    for entry in results:
        if entry["standard_type"] not in standard_types:
            continue
        if entry["relation"] not in relations:
            continue
        rows.append({
            "smiles":                 entry["canonical_smiles"],
            "standard_type":          entry["standard_type"],
            "value":                  entry["standard_value"],
            "units":                  entry["standard_units"],
            "assay_description":      entry["assay_description"],
            "assay_chembl_id":        entry["assay_chembl_id"],
            "molecule_chembl_id":     entry["molecule_chembl_id"],
            "document_chembl_id":     entry["document_chembl_id"],
            "data_validity_comment":  entry["data_validity_comment"],
        })

    return pd.DataFrame(rows)


def canonicalize_smiles(smi: str) -> Tuple[Optional[str], str]:
    """Convert SMILES to canonical form."""    
    if not smi or not isinstance(smi, str):
        return None, "fail: Invalid input SMILES"
    
    mol = MolFromSmiles(smi)
    if mol is None:
        return None, "fail: Cannot parse SMILES"
    
    try:
        smi_canon = MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return smi_canon, "pass"
    except Exception as e:
        return None, f"fail: Canonicalization error - {str(e)}"
    

def get_scaffold(smiles: str, scaffold_type: str = 'bemis_murcko') -> Mol:
    
    all_scaffs = ['bemis_murcko', 'generic', 'cyclic_skeleton']
    if scaffold_type not in all_scaffs:
        raise ValueError(f"scaffold_type='{scaffold_type}' is not supported. Pick from: {all_scaffs}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # designed to match atoms that are doubly bonded to another atom.
    PATT = Chem.MolFromSmarts("[$([D1]=[*])]")
    # replacement SMARTS (matches any atom)
    REPL = Chem.MolFromSmarts("[*]")

    Chem.RemoveStereochemistry(mol)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)

    if scaffold_type == 'generic':
        scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)

    elif scaffold_type == 'cyclic_skeleton':
        scaffold = AllChem.ReplaceSubstructs(scaffold, PATT, REPL, replaceAll=True)[0]
        scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        scaffold = MurckoScaffold.GetScaffoldForMol(scaffold)
    
    return scaffold


def calculate_ecfps(smiles: List[str], radius: int = 2, nbits: int = 2048, 
                   as_dict: bool = False) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Calculate ECFP fingerprints for multiple SMILES."""    
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)

    fps = [mfpgen.GetFingerprint(MolFromSmiles(smi)) for smi in smiles]
    return np.array(fps)


def is_common_solvent_fragment(smiles_frag: str) -> bool:
    """
    Return True if this standalone fragment is one of the known common solvents.
    Uses canonical SMILES matching.
    """
    mol = MolFromSmiles(smiles_frag)
    if mol is None:
        return False
    can = MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return can in COMMON_SOLVENTS


def canonicalize_smiles(smi: str) -> Tuple[Optional[str], str]:
    """
    Initial Canonicalization
    Convert SMILES to canonical form and validate parseability.
    """   
    mol = MolFromSmiles(smi)
    if mol is None:
        return None, "fail: Cannot parse SMILES"
    
    try:
        smi_canon = MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return smi_canon, "pass"
    except Exception as e:
        return None, f"fail: Canonicalization error - {str(e)}"


def remove_salts(smi: str, salt_smarts: str = SMARTS_COMMON_SALTS) -> Tuple[Optional[str], str]:
    """
    Salt Removal
    Remove common salt counterions.
    """   
    remover = SaltRemover(defnData=f"{salt_smarts}\tsalts")
    
    if '.' not in smi:
        return smi, "pass"
    
    try:
        mol = MolFromSmiles(smi)
        if mol is None:
            return None, "fail: Cannot parse SMILES"
        
        cleaned_mol = remover.StripMol(mol, dontRemoveEverything=True)
        
        if cleaned_mol is None or cleaned_mol.GetNumAtoms() == 0:
            return None, "fail: All fragments were salts"
        
        cleaned_smi = MolToSmiles(cleaned_mol)
        
        if '.' in cleaned_smi:
            frags = cleaned_smi.split('.')
            largest_frag = max(frags, key=len)
            return largest_frag, "pass"
        else:
            return cleaned_smi, "pass"
            
    except Exception as e:
        return None, f"fail: Salt removal error - {str(e)}"


def remove_solvents(smi: str) -> Tuple[Optional[str], str]:
    """
    Solvent Removal
    Strip common solvents from fragmented SMILES.
    """   
    if '.' not in smi:
        return smi, "pass"
    
    try:
        frags = [f.strip() for f in smi.split('.') if f.strip()]
        
        kept = []
        any_removed = False
        
        for frag in frags:
            if is_common_solvent_fragment(frag, COMMON_SOLVENTS):
                any_removed = True
            else:
                kept.append(frag)
        
        if not any_removed:
            return smi, "pass"
        
        if not kept:
            return None, "fail: All fragments were solvents"
        
        return '.'.join(kept), "pass"
        
    except Exception as e:
        return None, f"fail: Solvent removal error - {str(e)}"


def defragment_smiles(smi: str, keep_largest_fragment: bool = True) -> Tuple[Optional[str], str]:
    """
    Defragmentation
    Isolate largest molecular component.
    """
    if '.' not in smi:
        return smi, "pass"
    
    try:
        frags = [f.strip() for f in smi.split('.') if f.strip()]
        
        if len(frags) != len(set(frags)):
            return frags[0], "pass"
        
        if keep_largest_fragment:
            largest_frag = max(frags, key=len)
            return largest_frag, "pass"
        else:
            return smi, "pass"
            
    except Exception as e:
        return None, f"fail: Defragmentation error - {str(e)}"


def normalize_functional_groups(smi: str) -> Tuple[Optional[str], str]:
    """
    Functional Group Normalization
    Standardize nitro groups, N-oxides, azides, etc.
    """
    mol = MolFromSmiles(smi)
    if mol is None:
        return None, "fail: Cannot parse SMILES"
    
    try:
        mol = rdMolStandardize.Normalizer().normalize(mol)
        smi_norm = MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return smi_norm, "pass"
    except Exception as e:
        return None, f"fail: Normalization error - {str(e)}"


def reionize_smiles(smi: str) -> Tuple[Optional[str], str]:
    """
    Reionization
    Adjust charge distributions to chemically preferred forms.
    """   
    mol = MolFromSmiles(smi)
    if mol is None:
        return None, "fail: Cannot parse SMILES"
    
    try:
        mol = rdMolStandardize.Reionizer().reionize(mol)
        smi_reion = MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return smi_reion, "pass"
    except Exception as e:
        return None, f"fail: Reionization error - {str(e)}"


def neutralize_smiles(smi: str) -> Tuple[Optional[str], str]:
    """
    Charge Neutralization
    Convert charged species to neutral forms where appropriate.
    """   
    mol = MolFromSmiles(smi)
    if mol is None:
        return None, "fail: Cannot parse SMILES"
    
    try:

        neutralization_reactions = [(MolFromSmarts(x), MolFromSmiles(y, False)) 
                                    for x, y in SMARTS_NEUTRALIZATION_PATTERNS]

        for reactant, product in neutralization_reactions:
            while mol.HasSubstructMatch(reactant):
                rms = ReplaceSubstructs(mol, reactant, product)
                mol = rms[0]
        
        smiles = MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return smiles, "pass"
        
    except Exception as e:
        return None, f"fail: Neutralization error - {str(e)}"


def canonicalize_tautomers(smi: str) -> Tuple[Optional[str], str]:
    """
    Tautomer Canonicalization 
    Standardize to RDKit's canonical tautomer.
    WARNING: Can remove/change stereochemistry.
    """   
    mol = MolFromSmiles(smi)
    if mol is None:
        return None, "fail: Cannot parse SMILES"
    
    try:
        has_stereo_input = '@' in smi or '/' in smi or '\\' in smi
        
        can_mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        canonical_smiles = MolToSmiles(can_mol, canonical=True, isomericSmiles=True)
        
        has_stereo_output = '@' in canonical_smiles or '/' in canonical_smiles or '\\' in canonical_smiles
        
        if has_stereo_input and not has_stereo_output:
            return canonical_smiles, "pass (WARNING: Stereochemistry removed)"
        
        return canonical_smiles, "pass"
    except Exception as e:
        return None, f"fail: Tautomer canonicalization error - {str(e)}"


def flatten_stereochemistry(smi: str) -> Tuple[Optional[str], str]:
    """
    Stereochemistry Flattening
    Remove all stereochemical information.
    """
    
    mol = MolFromSmiles(smi)
    if mol is None:
        return None, "fail: Cannot parse SMILES"
    
    try:
        RemoveStereochemistry(mol)
        smi_flat = MolToSmiles(mol, isomericSmiles=False, canonical=True)
        if not smi_flat or smi_flat.strip() == "":
            return None, "fail: Stereochemistry flattening"
        return smi_flat, "pass"
    except Exception as e:
        return None, f"fail: Stereochemistry flattening error - {str(e)}"
    

def validate_smiles(smi: str) -> Tuple[Optional[str], str]:
    """
    Final Validation
    Verify the SMILES is valid and has at least one atom.
    """    
    try:
        mol = MolFromSmiles(smi)
    except Exception as e:
        return None, f"fail: Parsing exception - {str(e)}"
    
    if mol is None:
        return None, "fail: Cannot parse SMILES"
    
    if mol.GetNumAtoms() == 0:
        return None, "fail: Empty molecule (0 atoms)"
    
    return smi, "pass"


def cleaning_pipeline(smiles: list[str]) -> tuple[list[str], list[str]]:
    """

    - Canonicalization
    - Salt removal
    - Solvent removal
    - Defragmentation
    - Functional group normalization
    - Reionization
    - Charge neutralization
    - Tautomer canonicalization
    - Stereochemistry flattening
    - Final validation

    """
    # start cleaning the data
    standardized_smiles = []
    statuses = []

    # Suppress annoying RDKit warnings
    RDLogger.DisableLog('rdApp.*')
    
    for smi in smiles:
        annotations = []  # Track issues that need attention
        original_smi = smi  

        # Step 1: Canonicalize
        smi, status = canonicalize_smiles(smi)
        if status != "pass":
            standardized_smiles.append(None)
            statuses.append(status)
            continue
        
        # Step 2: Remove salts
        smi, status = remove_salts(smi)
        if status != "pass":
            standardized_smiles.append(None)
            statuses.append(status)
            continue
        
        # Step 3: Remove solvents
        smi, status = remove_solvents(smi)
        if status != "pass":
            standardized_smiles.append(None)
            statuses.append(status)
            continue
        
        # Step 4: Defragment
        smi, status = defragment_smiles(smi)
        if status != "pass":
            standardized_smiles.append(None)
            statuses.append(status)
            continue
        
        # Check if still fragmented after defragmentation
        if '.' in smi:
            annotations.append("still fragmented")
        
        # Step 5: Normalize functional groups
        smi, status = normalize_functional_groups(smi)
        if status != "pass":
            standardized_smiles.append(None)
            statuses.append(status)
            continue
        
        # Step 6: Reionize
        smi, status = reionize_smiles(smi)
        if status != "pass":
            standardized_smiles.append(None)
            statuses.append(status)
            continue
        
        # Step 7: Neutralize
        smi, status = neutralize_smiles(smi)
        if status != "pass":
            standardized_smiles.append(None)
            statuses.append(status)
            continue
        
        # Check if still charged after neutralization - annotate if any charged atoms
        mol_check = MolFromSmiles(smi)
        if mol_check is not None:
            charged_atoms = [atom for atom in mol_check.GetAtoms() if atom.GetFormalCharge() != 0]
            if charged_atoms:
                annotations.append("still has charged atoms after neutralization")
        
        # Step 8: Canonicalize tautomer
        smi, status = canonicalize_tautomers(smi)
        # Allow pass or pass with warnings to continue
        if not status.startswith("pass"):
            standardized_smiles.append(None)
            statuses.append(status)
            continue
        
        # Step 9: Flatten stereochemistry
        smi, status = flatten_stereochemistry(smi)
        if status != "pass":
            standardized_smiles.append(None)
            statuses.append(status)
            continue
        
        # Check for metals/metalloids
        mol_check = MolFromSmiles(smi)
        if mol_check is not None:
            metal_symbols = [
                'Li', 'Be', 'B', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 
                'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
                'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Cs', 'Ba', 'La', 'Ce',
                'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb',
                'Bi', 'Po', 'At', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U'
            ]
            for atom in mol_check.GetAtoms():
                if atom.GetSymbol() in metal_symbols:
                    annotations.append("contains metal/metalloid")
                    break
        
        # Step 12: Final validation
        smi, status = validate_smiles(smi)
        
        standardized_smiles.append(smi)
        
        # Add annotations to status if molecule passed
        if status == "pass" and annotations:
            statuses.append(f"pass: {', '.join(annotations)}")
        else:
            statuses.append(status)

    return standardized_smiles, statuses


def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = 'standardized_smiles',
    test_size: float = 0.2,
    scaffold_type: str = 'bemis_murcko',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Scaffold-split a DataFrame into train and test sets.

    Groups molecules by their Bemis-Murcko scaffold, then greedily assigns
    scaffold groups to test until the target fraction is met — so no scaffold
    appears in both splits. Scaffold groups are sorted largest-first for
    efficient packing. Ringless molecules (empty scaffold) always go to train.

    Args:
        df:            DataFrame containing at least the SMILES column.
        smiles_col:    Name of the column holding SMILES strings.
        test_size:     Fraction of molecules to place in the test set.
        scaffold_type: Passed to ``get_scaffold``; 'bemis_murcko' (default),
                       'generic', or 'cyclic_skeleton'.

    Returns:
        (train_df, test_df) — two DataFrames with reset indices.
    """
    from collections import defaultdict

    # Map scaffold SMILES → list of positional indices; ringless → None key
    scaffold_to_indices: dict[str | None, list[int]] = defaultdict(list)
    for pos, smi in enumerate(df[smiles_col].tolist()):
        scaffold_mol = get_scaffold(smi, scaffold_type=scaffold_type)
        if scaffold_mol is None or scaffold_mol.GetNumAtoms() == 0:
            scaffold_smi = None  # ringless — goes to train
        else:
            scaffold_smi = Chem.MolToSmiles(scaffold_mol)
        scaffold_to_indices[scaffold_smi].append(pos)

    # Ringless molecules always go to train
    train_idx: list[int] = list(scaffold_to_indices.pop(None, []))
    test_idx: list[int] = []

    # Assign remaining scaffold groups greedily, smallest first (rare/novel scaffolds → test)
    n_test_target = int(len(df) * test_size)
    groups = sorted(scaffold_to_indices.values(), key=len, reverse=False)
    for group in groups:
        if len(test_idx) < n_test_target:
            test_idx.extend(group)
        else:
            train_idx.extend(group)

    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

