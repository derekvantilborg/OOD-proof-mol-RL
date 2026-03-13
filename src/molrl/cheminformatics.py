# SMILES cleaning, canonicalisation, ECFP computation, Tanimoto similarity,
# scaffold splitting, medicinal chemistry filters.

import pandas as pd
from chembl_webresource_client.new_client import new_client


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

