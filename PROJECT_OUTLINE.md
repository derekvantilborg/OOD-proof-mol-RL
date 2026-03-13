# Unfamiliarity-Aware Molecular Optimization via Reinforcement Learning

## Motivation

Reinforcement learning (RL) for *de novo* molecular design typically fine-tunes a generative model (the *agent*) to produce molecules that score highly according to an oracle — a predictive model for a property of interest such as bioactivity. The fundamental vulnerability of this approach is **reward hacking**: the agent drifts into regions of chemical space where the oracle is unreliable, exploiting extrapolation artefacts rather than discovering genuinely active molecules. Standard diversity filters and KL penalties mitigate mode collapse but do not directly address oracle reliability.

The **Joint Molecular Model (JMM)** (van Tilborg, Rossen & Grisoni, *Nat. Mach. Intell.*, 2025; preprint: https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-qj4k3-v3; code: https://github.com/molML/JointMolecularModel) provides a principled solution. By jointly training a molecular property predictor and an autoencoder, the JMM produces an **unfamiliarity score** — the reconstruction loss — that quantifies how far a molecule lies from the oracle's training distribution. This score can be used to down-weight or filter oracle predictions on out-of-distribution (OOD) molecules during RL, preventing the agent from exploiting unreliable predictions.

This project tests that hypothesis in a controlled setting and serves as a learning exercise for JAX, Flax NNX, and reinforcement learning.

**Author context:** The project author developed the JMM and is experienced with PyTorch, LSTMs, and cheminformatics. JAX/Flax NNX and RL are new — the Transformer generator and RL loop are the primary learning targets. The JMM oracle is a reimplementation of familiar architecture in a new framework. Existing cheminformatics utility code (SMILES cleaning, ECFP computation, Tanimoto similarity, scaffold splitting) will be ported from previous projects into `src/cheminformatics.py`.


## Hypothesis

> An RL agent guided by an unfamiliarity-aware oracle (where predicted bioactivity scores are penalised in proportion to the molecule's unfamiliarity) will produce molecules that are (a) chemically more plausible, (b) closer to the oracle's reliable prediction domain, and (c) less susceptible to reward hacking — compared to an identical agent guided by an unweighted oracle.


## Target & Data

**Target:** EGFR (CHEMBL203) is the primary candidate — well-studied SAR, diverse scaffolds, clinically relevant. The final target will be confirmed after ChEMBL data wrangling, based on availability of sufficient Ki or EC50 measurements. Additional targets (e.g., JAK2, DRD2) can be added later to test generalisability.

**Bioactivity data considerations:**
- **Only Ki or EC50 values** — IC50s are excluded due to high inter-lab variability stemming from assay condition dependence
- Converted to pKi / pEC50 (−log10 of molar value) for regression
- Filtered to: single protein assay, exact measurements only (no qualifiers like `>` or `<`), deduplicated by canonical SMILES (median aggregation for duplicates)

**Data sources:**
- **Pretraining corpus:** A cleaned subset of ChEMBL SMILES (~1–2M molecules) — used for pretraining both the generator Transformer and the JMM autoencoder. **Filtered to exclude oracle test molecules and their close neighbours** (see below)
- **Oracle labelled data:** Filtered ChEMBL Ki/EC50 data for the selected target — used for pretraining the prediction head and for joint fine-tuning

**Preventing data leakage into the unfamiliarity signal:**

The unfamiliarity score is the reconstruction loss — if the autoencoder has seen a molecule during pretraining, it will reconstruct it well *regardless* of whether the oracle can reliably predict its activity. This means test molecules must be excluded from all pretraining data to keep the unfamiliarity validation honest.

Splitting procedure:
1. Scaffold-split the labelled EGFR data into train / validation / test
2. Remove all **test-set molecules and their close neighbours** (Tanimoto similarity > 0.4 on ECFP4) from the ChEMBL pretraining corpus
3. Train/validation molecules from the labelled set remain in the pretraining corpus — the oracle *should* be familiar with these
4. Use this filtered corpus for both AE pretraining and generator pretraining

This ensures that when the oracle assigns low unfamiliarity to a molecule, it's because the molecule is genuinely within the oracle's training domain — not because the AE memorised it during pretraining.


## Model Components

### 1. Molecular Generator (Prior / Agent)

A small **decoder-only Transformer** that autoregressively generates SMILES strings token-by-token. Implemented from scratch in JAX/Flax NNX as a learning exercise.

**Architecture:**
- Token embedding + learned positional embedding
- 4–6 Transformer decoder blocks (causal self-attention + feed-forward)
- Embedding dimension: 256, attention heads: 4–8, feed-forward dimension: 512–1024
- Vocabulary: character-level SMILES tokens (~70 tokens including start/end/pad)
- Output: softmax over vocabulary at each position

**Pretraining:**
- Teacher-forced next-token prediction on ChEMBL SMILES
- **Randomised SMILES augmentation is done at preprocessing time, not in the dataloader** — each SMILES is randomised ~5× using RDKit and all variants are stored in the HDF5 file. This avoids expensive RDKit calls during training (reassembling molecular graphs from token indices would be prohibitively slow). The pretraining corpus thus expands from ~1–2M to ~5–10M sequences on disk
- The pretrained model serves as both the **prior** (frozen, for KL regularisation) and the initial weights of the **agent** (fine-tuned during RL)

**Vocabulary:** A single shared character-level SMILES vocabulary (~70 tokens including start/end/pad) is used across all models (generator, AE encoder, AE decoder). Built once during data preparation and stored as `vocabulary.json`.

### 2. JMM Oracle (Simplified)

A simplified version of the JMM that produces two outputs for each molecule:
1. **Predicted pKi/pEC50** (regression) — the oracle score
2. **Unfamiliarity score** (reconstruction loss) — the distribution shift signal

**Architecture (modular — each component usable independently and jointly):**
- **Encoder:** Bidirectional LSTM that maps a SMILES token sequence → latent vector (faithful to the original JMM paper)
- **Decoder:** Unidirectional LSTM that maps latent vector → reconstructed SMILES (teacher-forced during training, greedy during inference). Also LSTM-based, matching the paper
- **Prediction head:** MLP that maps latent vector → pKi/pEC50 (single deterministic head; no Bayesian ensemble)
- **Unfamiliarity** = reconstruction loss (negative log-likelihood of the input SMILES given the latent representation)

**Training (three phases, following the paper):**

| Phase | What trains | Data | Purpose |
|-------|------------|------|---------|
| **1. AE pretraining** | Encoder + Decoder | Filtered ChEMBL SMILES (~1–2M, augmented ~5× to ~5–10M) | Learn a general molecular latent space from unlabelled data. Uses the same pre-augmented, pre-tokenised HDF5 corpus as generator pretraining |
| **2. Prediction head pretraining** | Prediction head (encoder frozen) | Target-specific labelled data (Ki/EC50) | Initialise the prediction head on top of the pretrained latent space |
| **3. Joint fine-tuning** | Encoder + Decoder + Prediction head (all unfrozen) | Target-specific labelled data | Align the latent space so it captures both structural and property-relevant features jointly |

Joint loss during phase 3: `L = α · prediction_loss (MSE) + (1 − α) · reconstruction_loss (CE on SMILES tokens)`

This phased approach is critical because the labelled target data (~3–10K molecules) is far too small to train a good autoencoder from scratch. Pretraining the AE on all of ChEMBL gives it a rich molecular representation, which is then specialised during joint fine-tuning.

**Design principle:** The encoder, decoder, and prediction head are implemented as independent Flax NNX modules that can be composed flexibly. This means the autoencoder can be used standalone (phase 1), the prediction head can be attached/detached (phase 2), and everything trains together (phase 3) without architectural surgery.

**Simplifications relative to the full JMM paper:**
- No Bayesian prediction ensemble — single deterministic prediction head
- No uncertainty quantification — only unfamiliarity (reconstruction loss) is used
- Regression on pKi/pEC50 instead of classification

### 3. RF-ECFP Baseline (Validation Only)

A **Random Forest** regressor on Morgan fingerprints (ECFP4, 2048 bits), trained on the same target data. Used solely to validate that the oracle task is non-trivial and that the JMM's predictive performance is reasonable. This model is **not** used in the RL loop.


## Reinforcement Learning Setup

### Approach: Augmented Likelihood (REINVENT-style)

The RL loop follows the established REINVENT paradigm adapted for this project:

1. **Sample** a batch of SMILES from the agent
2. **Score** each valid molecule with the oracle
3. **Update** the agent's policy to increase the likelihood of high-scoring molecules while staying close to the prior

**Policy gradient update (DAP-style):**

```
loss = -σ · score(x) · log P_agent(x) + log P_agent(x) - log P_prior(x)
```

Where:
- `P_agent(x)` = agent's likelihood of generating molecule x
- `P_prior(x)` = frozen prior's likelihood (KL regularisation term)
- `σ` = scalar controlling score influence
- `score(x)` = composite scoring function (see below)

### Scoring Functions (Experimental Conditions)

The composite score explicitly combines two signals: **bioactivity** (higher is better) and **unfamiliarity** (lower is better).

Two experimental conditions:

| Condition | Scoring function | Purpose |
|-----------|-----------------|---------|
| **Baseline (no weighting)** | `score(x) = pKi(x)` | Standard RL — expected to be susceptible to reward hacking |
| **Unfamiliarity-weighted** | `score(x) = pKi(x) · w(unfamiliarity(x))` | Core experiment — the unfamiliarity-aware oracle |

**Weighting function** `w(u)`:
- `w(u) = 1 / (1 + β · u)` where `u` is the unfamiliarity score (reconstruction loss) and `β` is a temperature parameter
- When `u ≈ 0` (molecule well-represented): `w → 1`, score ≈ raw pKi — the oracle's prediction is trusted
- When `u` is large (molecule far from training data): `w → 0`, score is dampened — the oracle's prediction is discounted
- `β` controls the sharpness of this transition

This keeps both components visible in a single scalar reward: the agent is explicitly rewarded for high predicted bioactivity *and* penalised for straying from the oracle's domain — all within the standard REINVENT loss formulation.

### RL Hyperparameters

| Parameter | Typical range | Notes |
|-----------|--------------|-------|
| Batch size | 64–128 | Molecules sampled per RL step |
| σ (score weight) | 50–128 | Controls exploitation vs. exploration |
| Learning rate | 1e-4 | Agent fine-tuning rate |
| RL steps | 500–2,000 | Total optimisation steps |
| β (unfamiliarity weight) | 1–10 | Temperature for weighting function (tune) |


## Experimental Design

### Phase 1: Data Preparation
1. Download and filter ChEMBL data for EGFR (CHEMBL203) — Ki/EC50 only
2. Clean and canonicalise SMILES; remove duplicates and salts
3. **Scaffold-split** labelled data into train / validation / test
4. Prepare ChEMBL pretraining corpus: clean full ChEMBL SMILES, **remove test-set molecules and neighbours** (Tanimoto > 0.4 on ECFP4), **augment ~5× with randomised SMILES**, tokenise, and save as HDF5
5. Apply the same augment → tokenise → HDF5 pipeline to the oracle train/val/test splits (with labels)

### Phase 2: Model Training
1. **Pretrain the autoencoder** (JMM encoder + decoder) on the filtered ChEMBL corpus
2. **Pretrain the Transformer generator** on the same filtered ChEMBL corpus (next-token prediction)
3. **Pretrain the JMM prediction head** on the target-specific labelled data (encoder frozen)
4. **Joint fine-tune the full JMM** (encoder + decoder + prediction head, all unfrozen) on the target-specific labelled data
5. **Train the RF-ECFP baseline** on the same target data
6. **Validate oracle quality:** Compare JMM and RF predictive performance (R², RMSE) on the held-out test set. Verify that unfamiliarity scores are higher for test-set molecules that are structurally dissimilar to the training set (sanity check)

### Phase 3: Hyperparameter Optimisation
- **JMM oracle:** Optimise latent dimension, α (loss balance), learning rate, and joint fine-tuning schedule using Optuna (TPE sampler + MedianPruner). Target metric: validation R² on pKi/pEC50 prediction
- **Generator:** Use literature-standard values; manual tuning only
- **RL loop:** Manual tuning of σ, learning rate, batch size (fast to iterate). Small grid search for β

### Phase 4: RL Experiments
1. Run both scoring conditions (baseline, unfamiliarity-weighted) with identical initialisations (same pretrained prior, same random seeds)
2. Run each condition with 3–5 random seeds for statistical robustness
3. Log all generated molecules, scores, and unfamiliarity values at every RL step

### Phase 5: Analysis
Compare the two conditions across the following metrics (tracked over RL steps):

**Generation quality:**
- Fraction of valid, unique, and novel SMILES
- Internal diversity (Tanimoto distance within generated batches)

**Oracle exploitation:**
- Distribution of predicted pKi/pEC50 scores over RL steps
- Top-K predicted pKi/pEC50 values achieved

**Distribution shift (the key analysis):**
- Distribution of unfamiliarity scores over RL steps — does the baseline drift to high unfamiliarity while the weighted condition stays grounded?
- Mean Tanimoto similarity to oracle training set over RL steps
- t-SNE or UMAP visualisations of generated molecules in the JMM's latent space, coloured by RL step and by condition

**Chemical plausibility:**
- QED (quantitative estimate of drug-likeness) distribution
- SA score (synthetic accessibility) distribution
- Fraction of molecules passing basic medicinal chemistry filters (no PAINS, Lipinski-compliant)

**Reward hacking detection:**
- Correlation between predicted pKi/pEC50 and unfamiliarity in the baseline condition (positive correlation = reward hacking)
- Comparison of "best" molecules from each condition: are the top-scoring baseline molecules chemically unreasonable?


## Tech Stack

| Component | Library | Notes |
|-----------|---------|-------|
| Core framework | JAX + Flax NNX | NNX is the new recommended Flax API; Pythonic mutable state |
| Optimisers | Optax | Standard JAX optimiser library |
| Checkpointing | Orbax | JAX-native serialisation |
| Hyperparameter optimisation | Optuna | TPE sampler + MedianPruner; SQLite storage for dashboard |
| Classical ML baseline | scikit-learn | Random Forest regressor |
| Cheminformatics | RDKit | SMILES processing, fingerprints, molecular descriptors |
| Data preprocessing | Polars / Pandas | One-time ChEMBL wrangling and filtering; Pandas where RDKit interop is needed |
| Training data loading | Grain + HDF5 (h5py) | Chunked reads from disk; Grain handles shuffling, batching, prefetching |
| Experiment tracking | Weights & Biases | Logging RL curves, generated molecules, metrics |
| Visualisation | plotnine | Python ggplot2 clone for publication-quality figures |
| Hardware | Apple M4 MacBook Pro | JAX MPS backend via [jax-mps](https://github.com/tillahoffmann/jax-mps) (MLX-backed PJRT plugin); budget ~1 day for generator pretraining |

### Data Loading Strategy

Training data loading is on the critical path for pretraining (~1–2M sequences, many epochs). On Apple Silicon, CPU and GPU share unified memory — loading the entire dataset as a NumPy array competes directly with model weights and activations. HDF5 with chunked reads solves this.

**Preprocessing (once, in the data preparation notebook):**
1. Clean and canonicalise all SMILES
2. **Augment:** generate ~5 randomised SMILES variants per molecule using RDKit (`MolToSmiles(mol, doRandom=True)`) — this is the expensive step, done once and saved
3. Tokenise all SMILES (canonical + augmented) → integer sequences (vocabulary indices)
4. Pad to a fixed max length
5. Save as HDF5 (`.h5`) with named datasets, chunked along the sample axis:
   - `"sequences"`: int16, shape `(N, max_len)`, chunked as `(batch_size, max_len)`
   - `"labels"`: float32, shape `(N,)` — present for oracle data, absent for pretraining corpus
   - One file format for both use cases
   - Pretraining corpus: ~5–10M augmented sequences (~1.5–3GB at int16 × 150 tokens)

**Training-time dataloader (Grain + h5py):**
1. `grain.RandomAccessDataSource` reads from the HDF5 file via h5py — only the requested batch is loaded into memory, not the full dataset
2. Grain handles shuffling, batching, and prefetching (next batch loads while current batch is on GPU)
3. Batches are cast to `jnp.array` at consumption time

This keeps memory footprint proportional to batch size, not dataset size — critical for fitting both data and model in the M4's unified memory pool.

For the **RL loop** (64–128 molecules per step), SMILES are generated by the agent and tokenised on-the-fly before passing to the oracle. Batch sizes are small enough that this adds negligible overhead.


## Compute Considerations

Running on an M4 MacBook Pro imposes constraints:

- **Generator pretraining:** The most expensive step. Consider a subset of ChEMBL (~500K–1M SMILES) if full training is too slow. With a small Transformer (4 layers, 256-dim), this should be feasible overnight
- **JMM training:** Relatively fast since the target-specific dataset is much smaller (~3K–10K molecules). Minutes to low hours
- **RL loop:** Each step involves sampling ~64–128 molecules and scoring them. The bottleneck is SMILES generation (autoregressive, sequential). Expect ~1–2 hours per full RL run (500–2000 steps). Running 2 conditions × 3–5 seeds = 6–10 runs total, so budget ~1 day for all RL experiments
- **Optuna HPO:** Focus HPO budget on the JMM oracle. Use ~20–50 trials with pruning. RL hyperparameters tuned manually


## Project Structure

```
unfamiliar-rl/
│
├── README.md
├── pyproject.toml
│
├── src/molrl/                        # Installable package (workhorse code)
│   ├── __init__.py
│   ├── nnx_modules.py                # Flax NNX building blocks: attention, LSTM encoder,
│   │                                 #   LSTM decoder, MLP head, positional embedding, etc.
│   ├── models.py                     # Full composed models: Generator (Transformer),
│   │                                 #   Autoencoder (enc+dec), JMM (enc+dec+head)
│   ├── cheminformatics.py            # SMILES cleaning, canonicalisation, ECFP computation,
│   │                                 #   Tanimoto similarity, scaffold splitting, filters
│   ├── data.py                       # Vocabulary, tokeniser, HDF5 Grain DataSource,
│   │                                 #   preprocessing utilities (SMILES → integer-encoded .h5)
│   ├── training.py                   # Training loops (pretraining, joint fine-tuning),
│   │                                 #   Optuna objective functions
│   ├── rl.py                         # RL loop, scoring functions (baseline & weighted),
│   │                                 #   REINVENT-style policy update, SMILES sampling
│   ├── eval.py                       # Metrics (R², RMSE, validity, uniqueness, novelty),
│   │                                 #   chemical quality (QED, SA, PAINS), reward hacking detection
│   └── utils.py                      # Config loading, checkpointing (Orbax), W&B logging,
│                                     #   reproducibility (seed management), path helpers
│
├── configs/                          # YAML config files
│   ├── pretrain_ae.yaml
│   ├── pretrain_generator.yaml
│   ├── oracle.yaml
│   └── rl.yaml
│
└── project/                          # Step-by-step execution (the "story" of the project)
    │
    ├── step_1_data_preparation/
    │   ├── fetch_and_clean.ipynb      # Download ChEMBL, filter Ki/EC50, clean SMILES
    │   ├── chembl_raw.csv             # Raw download (gitignored)
    │   ├── chembl_smiles_clean.csv    # Cleaned full ChEMBL SMILES
    │   ├── egfr_labelled.csv          # Filtered EGFR Ki/EC50 data
    │   ├── egfr_train.csv             # ┐
    │   ├── egfr_val.csv               # ├ Scaffold split
    │   ├── egfr_test.csv              # ┘
    │   ├── pretrain_corpus.h5         # Filtered ChEMBL (test neighbours removed), integer-encoded
    │   ├── oracle_train.h5            # EGFR train split, integer-encoded + labels
    │   ├── oracle_val.h5              # EGFR val split, integer-encoded + labels
    │   ├── oracle_test.h5             # EGFR test split, integer-encoded + labels
    │   └── vocabulary.json            # Token ↔ index mapping
    │
    ├── step_2_pretrain_autoencoder/
    │   ├── pretrain_ae.ipynb           # AE pretraining on filtered ChEMBL corpus
    │   └── checkpoints/               # Orbax checkpoints for pretrained encoder + decoder
    │
    ├── step_3_pretrain_generator/
    │   ├── pretrain_transformer.ipynb  # Transformer pretraining on filtered ChEMBL corpus
    │   └── checkpoints/               # Orbax checkpoints for pretrained prior
    │
    ├── step_4_train_oracle/
    │   ├── train_oracle.ipynb          # Prediction head pretraining + joint fine-tuning + HPO
    │   ├── train_rf_baseline.ipynb     # RF-ECFP baseline for oracle validation
    │   ├── optuna.db                   # Optuna study (SQLite)
    │   └── checkpoints/               # Final JMM oracle checkpoint
    │
    ├── step_5_validate_oracle/
    │   ├── validate_oracle.ipynb       # Oracle quality: R², RMSE vs RF; unfamiliarity sanity check
    │   └── figures/                    # Validation plots
    │
    ├── step_6_rl_experiments/
    │   ├── run_rl.ipynb                # Launch baseline + weighted RL runs (multiple seeds)
    │   ├── runs/                       # Per-run outputs: generated molecules, scores, unfamiliarity
    │   │   ├── baseline_seed0/
    │   │   ├── baseline_seed1/
    │   │   ├── weighted_seed0/
    │   │   └── ...
    │   └── checkpoints/               # Agent checkpoints per run
    │
    └── step_7_analysis/
        ├── analysis.ipynb              # Full comparative analysis across conditions
        └── figures/                    # Publication-quality plotnine figures
```

### Design Principles

**`src/molrl/` is the reusable library.** All logic lives here as importable functions and classes. No state, no paths, no side effects at import time. Every function takes its dependencies as arguments.

**`project/` is the executed narrative.** Each step directory is self-contained: one or two notebooks plus all the artefacts produced at that step. An outsider reads step 1 → 2 → ... → 7 and sees the full pipeline. Dependencies between steps are explicit — a notebook in step 4 loads a checkpoint from `../step_2_pretrain_autoencoder/checkpoints/`.

**`nnx_modules.py` vs `models.py`:** Modules are the building blocks (attention block, LSTM encoder, LSTM decoder, MLP head). Models are compositions of modules into full architectures (Generator wraps a Transformer, Autoencoder wraps encoder + decoder, JMM wraps Autoencoder + prediction head). This supports phased training naturally — you can freeze/unfreeze at the module level, and the Autoencoder exists as a usable model independent of the JMM.

**Configs are separate from code.** Each training step reads a YAML config that specifies hyperparameters, paths, and training settings. The notebooks load the config and pass it to functions from `src/`. This means re-running with different hyperparameters doesn't require editing code.


## Scope: MVP vs Future Extensions

**This project is an MVP.** The goal is to prove the concept with the simplest viable setup, then expand if results are promising.

**In scope (MVP):**
- One target (EGFR)
- One unfamiliarity integration method (soft weighting)
- Two RL conditions (baseline vs. weighted)
- Single deterministic JMM (no Bayesian ensemble, no uncertainty quantification)
- Small Transformer generator, LSTM-based JMM

**Future extensions (out of scope for now):**
- Additional targets (JAK2, DRD2) to test generalisability
- Hard-filtering scoring variant (`score = 0` if `unfamiliarity > τ`)
- Two separate reward terms in the RL loss instead of a composite score
- Bayesian prediction ensemble for calibrated uncertainty
- Larger generator architectures or graph-based representations
- Wet-lab validation of top-scoring molecules


## Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Generator pretraining too slow on M4 | Use smaller ChEMBL subset; reduce model size; use mixed precision |
| JMM oracle not predictive enough | Validate against RF-ECFP baseline; ensure sufficient training data |
| Unfamiliarity score not discriminative | Sanity-check on held-out OOD molecules before running RL |
| RL doesn't converge / mode collapses | Start with literature σ values; ensure prior KL penalty is active |
| Unfamiliarity weighting too aggressive → agent stuck | Tune β carefully; start with small values and increase; can fall back to hard filtering as an alternative |
| jax-mps MPS backend instability on Apple Silicon | Pin jax + jaxlib to 0.9.x (matching jax-mps build); fall back to CPU if needed (slower but stable) |


## Success Criteria

The project is successful if:

1. The baseline RL agent demonstrably exhibits reward hacking (high predicted pKi/pEC50 but also high unfamiliarity, low similarity to training set, chemically questionable molecules)
2. The unfamiliarity-weighted agent produces molecules with comparable or slightly lower predicted pKi/pEC50 but significantly lower unfamiliarity and higher chemical plausibility
3. The analysis clearly shows the divergence between conditions in the unfamiliarity distribution over RL steps
4. The codebase is clean, well-structured, and serves as a useful reference for JAX-based RL molecular design
