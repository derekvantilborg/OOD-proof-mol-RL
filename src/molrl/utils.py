# Config loading, checkpointing (Orbax), W&B logging, reproducibility
# (seed management), path helpers.

from flax import nnx
import jax
import jax.numpy as jnp
from molrl.tokenizer import encoding_to_smiles


@nnx.jit
def _sample_tokens_compiled(model, tokens, key, t, eos_id, pad_id):
    """Compiled autoregressive sampling with EOS-aware padding per sequence."""
    batch_size, seq_len = tokens.shape
    done0 = jnp.zeros((batch_size,), dtype=jnp.bool_)

    def body_fn(pos, state):
        current_tokens, done, current_key = state
        logits = model(current_tokens, is_training=False)
        next_logits = logits[:, pos, :] / t

        current_key, subkey = jax.random.split(current_key)
        sampled = jax.random.categorical(subkey, next_logits, axis=-1).astype(jnp.int32)

        # Once EOS is emitted, keep writing PAD.
        next_tokens = jnp.where(done, jnp.asarray(pad_id, dtype=jnp.int32), sampled)
        updated_tokens = current_tokens.at[:, pos + 1].set(next_tokens)
        updated_done = jnp.logical_or(done, next_tokens == eos_id)

        return updated_tokens, updated_done, current_key

    final_tokens, _, _ = jax.lax.fori_loop(0, seq_len - 1, body_fn, (tokens, done0, key))
    return final_tokens


def transformer_generate_smiles(model, max_seq_len: int, t: float, n: int, eos_id=35, pad_id=0):
    """Generate n SMILES strings using autoregressive temperature sampling.

    Args:
        model: Trained AutoregressiveTransformer.
        max_seq_len: Maximum sequence length.
        t: Sampling temperature (> 0).
        n: Number of sequences to sample.
        eos_id: End-of-sequence token id (default 35).
        pad_id: Padding token id (default 0).

    Returns:
        jnp.ndarray of shape [n, max_seq_len] with integer token ids.
    """
    if t <= 0:
        raise ValueError("t must be > 0")
    if n <= 0:
        raise ValueError("n must be > 0")

    max_len = int(max_seq_len)
    tokens = jnp.full((n, max_len), pad_id, dtype=jnp.int32)
    tokens = tokens.at[:, 0].set(1)

    import os
    seed = int.from_bytes(os.urandom(4), "little")
    key = jax.random.PRNGKey(seed)

    designs = _sample_tokens_compiled(
        model,
        tokens,
        key,
        jnp.asarray(t, dtype=jnp.float32),
        jnp.asarray(eos_id, dtype=jnp.int32),
        jnp.asarray(pad_id, dtype=jnp.int32),
    )

    smiles = [encoding_to_smiles(designs[i].tolist()) for i in range(len(designs))]

    return smiles
