from typing import Optional

from flax import nnx
import jax
import jax.numpy as jnp
from .nnx_modules import TransformerBlock

class AutoregressiveTransformer(nnx.Module):
	"""Simple causal language model style Transformer in Flax NNX."""

	def __init__(
		self,
		vocab_size: int,
		max_seq_len: int,
		emb_dim: int = 256,
		num_layers: int = 4,
		num_heads: int = 8,
		mlp_dim: int = 1024,
		dropout_rate: float = 0.1,
		rngs: Optional[nnx.Rngs] = None,
	):
		if rngs is None:
			rngs = nnx.Rngs(0)

		embed_init = jax.nn.initializers.normal(stddev=0.02)
		kernel_init = jax.nn.initializers.normal(stddev=0.02)

		self.vocab_size = vocab_size
		self.max_seq_len = max_seq_len
		self.emb_dim = emb_dim

		self.token_embedding = nnx.Embed(num_embeddings=vocab_size, features=emb_dim, embedding_init=embed_init, rngs=rngs)
		self.position_embedding = nnx.Embed(num_embeddings=max_seq_len, features=emb_dim, embedding_init=embed_init, rngs=rngs)
		self.transformer_blocks = nnx.List([TransformerBlock(emb_dim=emb_dim, num_heads=num_heads, mlp_dim=mlp_dim, dropout_rate=dropout_rate, rngs=rngs) for _ in range(num_layers)])
		self.output_norm = nnx.LayerNorm(num_features=emb_dim, rngs=rngs)
		self.output_projection = nnx.Linear(in_features=emb_dim, out_features=vocab_size, kernel_init=kernel_init, rngs=rngs)

	def __call__(self, input_ids: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = None, is_training: bool = False) -> jnp.ndarray:
		
		deterministic = not is_training
		
		if input_ids.ndim != 2:
			raise ValueError("input_ids must have shape [batch, seq_len]")

		batch_size, seq_len = input_ids.shape
		if seq_len > self.max_seq_len:
			raise ValueError(f"seq_len ({seq_len}) exceeds max_seq_len ({self.max_seq_len})")

		token_embed = self.token_embedding(input_ids)
		pos_ids = jnp.arange(seq_len)[None, :]
		pos_embed = self.position_embedding(pos_ids)

		x = token_embed + pos_embed

		causal_mask = nnx.make_causal_mask(jnp.ones((batch_size, seq_len), dtype=jnp.bool_))
		if attention_mask is not None:
			pad_mask = nnx.make_attention_mask(
				attention_mask.astype(jnp.bool_),
				attention_mask.astype(jnp.bool_),
			)
			attn_mask = nnx.combine_masks(causal_mask, pad_mask)
		else:
			attn_mask = causal_mask

		for block in self.transformer_blocks:
			x = block(x, attn_mask=attn_mask, deterministic=deterministic)

		x = self.output_norm(x)
		return self.output_projection(x)
