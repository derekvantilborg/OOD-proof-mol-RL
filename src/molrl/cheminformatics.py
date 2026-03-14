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
from tqdm.auto import tqdm

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


def _ecfp_chunk_worker(smiles_chunk: List[str], radius: int, nbits: int) -> np.ndarray:
    """Process a chunk of SMILES — generator created once per chunk."""
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    fps = []
    for smi in smiles_chunk:
        mol = MolFromSmiles(smi)
        fps.append(np.array(mfpgen.GetFingerprint(mol)) if mol is not None else np.zeros(nbits, dtype=np.uint8))
    return np.array(fps)


def calculate_ecfps(smiles: List[str], radius: int = 2, nbits: int = 2048,
                    n_jobs: int = -1, chunk_size: int = 100) -> np.ndarray:
    """Calculate ECFP fingerprints for multiple SMILES using multiprocessing."""
    from multiprocessing import Pool, cpu_count
    from functools import partial

    chunks = [smiles[i:i + chunk_size] for i in range(0, len(smiles), chunk_size)]
    n_workers = cpu_count() if n_jobs == -1 else n_jobs
    worker = partial(_ecfp_chunk_worker, radius=radius, nbits=nbits)
    with Pool(processes=n_workers) as pool:
        results = pool.map(worker, chunks)
    return np.concatenate(results, axis=0)


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


def normalize_functional_groups(smi: str, func_group_normalizer: rdMolStandardize.Normalizer = None) -> Tuple[Optional[str], str]:
    """
    Functional Group Normalization
    Standardize nitro groups, N-oxides, azides, etc.
    """
    mol = MolFromSmiles(smi)
    if mol is None:
        return None, "fail: Cannot parse SMILES"
    
    try:
        if func_group_normalizer is None:
            func_group_normalizer = rdMolStandardize.Normalizer()
        mol = func_group_normalizer.normalize(mol)
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


def neutralize_smiles(smi: str, neutralization_reactions = None) -> Tuple[Optional[str], str]:
    """
    Charge Neutralization
    Convert charged species to neutral forms where appropriate.
    """   
    mol = MolFromSmiles(smi)
    if mol is None:
        return None, "fail: Cannot parse SMILES"
    
    try:

        if neutralization_reactions is None:
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


def canonicalize_tautomers(smi: str, tautomer_enumerator: rdMolStandardize.TautomerEnumerator = None) -> Tuple[Optional[str], str]:
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
        
        if tautomer_enumerator is None:
            tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
        
        can_mol = tautomer_enumerator.Canonicalize(mol)
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


def _cleaning_pipeline_chunk_worker(args: tuple) -> tuple[list, list]:
    """Process a chunk of SMILES through the full cleaning pipeline.
    
    Takes a tuple (smiles_chunk, skip_tautomer_canonicalization) so it is
    compatible with pool.imap. All RDKit stateful objects are created once
    per chunk to avoid pickling issues across process boundaries.
    """
    from rdkit.Chem.MolStandardize import rdMolStandardize as _rdMolStd

    smiles_chunk, skip_tautomer_canonicalization = args

    RDLogger.DisableLog('rdApp.*')

    # Create once per chunk
    tautomer_enumerator = _rdMolStd.TautomerEnumerator()
    neutralization_reactions = [(MolFromSmarts(x), MolFromSmiles(y, False))
                                for x, y in SMARTS_NEUTRALIZATION_PATTERNS]
    func_group_normalizer = _rdMolStd.Normalizer()

    standardized_smiles = []
    statuses = []

    for smi in smiles_chunk:
        annotations = []

        smi, status = canonicalize_smiles(smi)
        if status != "pass":
            standardized_smiles.append(None); statuses.append(status); continue

        smi, status = remove_salts(smi)
        if status != "pass":
            standardized_smiles.append(None); statuses.append(status); continue

        smi, status = remove_solvents(smi)
        if status != "pass":
            standardized_smiles.append(None); statuses.append(status); continue

        smi, status = defragment_smiles(smi)
        if status != "pass":
            standardized_smiles.append(None); statuses.append(status); continue

        if '.' in smi:
            annotations.append("still fragmented")

        smi, status = normalize_functional_groups(smi, func_group_normalizer)
        if status != "pass":
            standardized_smiles.append(None); statuses.append(status); continue

        smi, status = reionize_smiles(smi)
        if status != "pass":
            standardized_smiles.append(None); statuses.append(status); continue

        smi, status = neutralize_smiles(smi, neutralization_reactions)
        if status != "pass":
            standardized_smiles.append(None); statuses.append(status); continue

        mol_check = MolFromSmiles(smi)
        if mol_check is not None:
            if any(atom.GetFormalCharge() != 0 for atom in mol_check.GetAtoms()):
                annotations.append("still has charged atoms after neutralization")

        if not skip_tautomer_canonicalization:
            smi, status = canonicalize_tautomers(smi, tautomer_enumerator)
            if not status.startswith("pass"):
                standardized_smiles.append(None); statuses.append(status); continue

        smi, status = flatten_stereochemistry(smi)
        if status != "pass":
            standardized_smiles.append(None); statuses.append(status); continue

        mol_check = MolFromSmiles(smi)
        if mol_check is not None:
            metal_symbols = {
                'Li', 'Be', 'B', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn',
                'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
                'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Cs', 'Ba', 'La', 'Ce',
                'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb',
                'Bi', 'Po', 'At', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U'
            }
            if any(atom.GetSymbol() in metal_symbols for atom in mol_check.GetAtoms()):
                annotations.append("contains metal/metalloid")

        smi, status = validate_smiles(smi)
        standardized_smiles.append(smi)
        statuses.append(f"pass: {', '.join(annotations)}" if status == "pass" and annotations else status)

    return standardized_smiles, statuses


def cleaning_pipeline(
    smiles: list[str],
    skip_tautomer_canonicalization: bool = False,
    n_jobs: int = -1,
    chunk_size: int = 100,
) -> tuple[list[str], list[str]]:
    """
    - Canonicalization
    - Salt removal
    - Solvent removal
    - Defragmentation
    - Functional group normalization
    - Reionization
    - Charge neutralization
    - Tautomer canonicalization  (skip with skip_tautomer_canonicalization=True)
    - Stereochemistry flattening
    - Final validation
    """
    from multiprocessing import Pool, cpu_count

    chunks = [
        (smiles[i:i + chunk_size], skip_tautomer_canonicalization)
        for i in range(0, len(smiles), chunk_size)
    ]
    n_workers = cpu_count() if n_jobs == -1 else n_jobs

    all_smiles: list[str] = []
    all_statuses: list[str] = []
    with Pool(processes=n_workers) as pool:
        for chunk_smiles, chunk_statuses in tqdm(
            pool.imap(_cleaning_pipeline_chunk_worker, chunks),
            total=len(chunks),
            desc="Cleaning SMILES",
        ):
            all_smiles.extend(chunk_smiles)
            all_statuses.extend(chunk_statuses)

    return all_smiles, all_statuses


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


def augment_smiles(smiles: str, n: int = 5, max_attempts: int = 100) -> Optional[str]:

    """Generate a random SMILES variant of the same molecule.

    Uses RDKit's random SMILES generation, which shuffles atom order and
    applies random kekulization. Returns None if no valid variant is found
    after max_attempts.

    Args:
        smiles:      Input SMILES string to augment.
        n:           Number of random variants to generate (default: 5).
        max_attempts: Maximum attempts to find a valid variant (default: 100).
    """
    mol = MolFromSmiles(smiles)
    if mol is None:
        return None
    
    new_smiles_set = set()

    for _ in range(max_attempts):
        random_smi = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
        if random_smi and random_smi != smiles:
            new_smiles_set.add(random_smi)
            if len(new_smiles_set) >= n:
                break

    if new_smiles_set:
        return list(new_smiles_set)
    else:
        return None


