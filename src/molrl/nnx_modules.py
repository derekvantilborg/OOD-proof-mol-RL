from flax import nnx
import jax
import jax.numpy as jnp


class TransformerBlock(nnx.Module):
	"""Single pre-norm Transformer decoder block in Flax NNX."""

	def __init__(
		self,
		emb_dim: int,
		num_heads: int,
		mlp_dim: int,
		dropout_rate: float,
		rngs: nnx.Rngs,
	):
		kernel_init = jax.nn.initializers.normal(stddev=0.02)

		self.attention_norm = nnx.LayerNorm(num_features=emb_dim, rngs=rngs)
		self.self_attention = nnx.MultiHeadAttention(
			num_heads=num_heads,
			in_features=emb_dim,
			qkv_features=emb_dim,
			out_features=emb_dim,
			dropout_rate=dropout_rate,
			decode=False,
			kernel_init=kernel_init,
			rngs=rngs,
		)
		self.attention_dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

		self.feed_forward_norm = nnx.LayerNorm(num_features=emb_dim, rngs=rngs)
		self.feed_forward_input = nnx.Linear(
			in_features=emb_dim,
			out_features=mlp_dim,
			kernel_init=kernel_init,
			rngs=rngs,
		)
		self.feed_forward_output = nnx.Linear(
			in_features=mlp_dim,
			out_features=emb_dim,
			kernel_init=kernel_init,
			rngs=rngs,
		)
		self.feed_forward_dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

	def __call__(
		self,
		x: jnp.ndarray,
		attn_mask: jnp.ndarray,
		deterministic: bool = True,
	) -> jnp.ndarray:
		y = self.attention_norm(x)
		y = self.self_attention(y, y, mask=attn_mask, deterministic=deterministic)
		y = self.attention_dropout(y, deterministic=deterministic)
		x = x + y

		y = self.feed_forward_norm(x)
		y = self.feed_forward_input(y)
		y = nnx.gelu(y)
		y = self.feed_forward_dropout(y, deterministic=deterministic)
		y = self.feed_forward_output(y)
		y = self.feed_forward_dropout(y, deterministic=deterministic)
		return x + y
