from typing import Optional, Sequence

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


# ---------------------------------------------------------------------------
# Joint Molecular Model (JMM) building blocks
# ---------------------------------------------------------------------------

class SmilesEncoder(nnx.Module):
	"""1D CNN encoder: integer-encoded SMILES → latent vector Z.

	Architecture: token embedding → stacked Conv1D + BatchNorm + ReLU →
	global max-pool → dropout → linear projection to latent_dim.
	"""

	def __init__(self, vocab_size: int, latent_dim: int = 128, emb_dim: int = 64,
				 conv_channels: Sequence[int] = (128, 256, 512),
				 kernel_sizes: Sequence[int] = (9, 9, 11),
				 dropout_rate: float = 0.2, rngs: Optional[nnx.Rngs] = None):
		if rngs is None:
			rngs = nnx.Rngs(0)

		kernel_init = jax.nn.initializers.normal(stddev=0.02)

		self.vocab_size = vocab_size
		self.latent_dim = latent_dim

		self.embedding = nnx.Embed(num_embeddings=vocab_size, features=emb_dim, rngs=rngs)

		self.conv_layers = nnx.List()
		self.conv_norms = nnx.List()
		in_ch = emb_dim
		for out_ch, ks in zip(conv_channels, kernel_sizes):
			self.conv_layers.append(nnx.Conv(in_features=in_ch, out_features=out_ch, kernel_size=(ks,), kernel_init=kernel_init, rngs=rngs))
			self.conv_norms.append(nnx.BatchNorm(num_features=out_ch, rngs=rngs))
			in_ch = out_ch

		self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
		self.fc = nnx.Linear(in_features=conv_channels[-1], out_features=latent_dim, kernel_init=kernel_init, rngs=rngs)

	def __call__(self, input_ids: jnp.ndarray, is_training: bool = False) -> jnp.ndarray:
		"""Encode token ids to latent vector.

		Args:
			input_ids: [batch, seq_len] integer token ids.
			is_training: enables dropout / batch-norm training mode.

		Returns:
			z: [batch, latent_dim]
		"""
		deterministic = not is_training
		x = self.embedding(input_ids)  # [batch, seq_len, emb_dim]

		for conv, norm in zip(self.conv_layers, self.conv_norms):
			x = conv(x)
			x = norm(x, use_running_average=deterministic)
			x = nnx.relu(x)

		x = jnp.max(x, axis=1)  # global max-pool → [batch, conv_channels[-1]]
		x = self.dropout(x, deterministic=deterministic)
		return self.fc(x)  # [batch, latent_dim]


class SmilesDecoder(nnx.Module):
	"""GRU decoder: latent vector Z → SMILES token logits.

	The latent vector is projected through a linear layer + tanh to produce
	the initial GRU hidden state.  Decoding uses teacher forcing (ground-truth
	tokens fed at each time step) during training.
	"""

	def __init__(self, vocab_size: int, max_seq_len: int, latent_dim: int = 128,
				 hidden_dim: int = 512, emb_dim: int = 64,
				 rngs: Optional[nnx.Rngs] = None):
		if rngs is None:
			rngs = nnx.Rngs(0)

		kernel_init = jax.nn.initializers.normal(stddev=0.02)

		self.vocab_size = vocab_size
		self.max_seq_len = max_seq_len
		self.hidden_dim = hidden_dim

		self.embedding = nnx.Embed(num_embeddings=vocab_size, features=emb_dim, rngs=rngs)
		self.z_to_hidden = nnx.Linear(in_features=latent_dim, out_features=hidden_dim, kernel_init=kernel_init, rngs=rngs)
		self.gru_cell = nnx.GRUCell(in_features=emb_dim, hidden_features=hidden_dim, kernel_init=kernel_init, recurrent_kernel_init=kernel_init, rngs=rngs)
		self.output_projection = nnx.Linear(in_features=hidden_dim, out_features=vocab_size, kernel_init=kernel_init, rngs=rngs)

	def __call__(self, z: jnp.ndarray, target_ids: jnp.ndarray) -> jnp.ndarray:
		"""Teacher-forced decoding.

		Args:
			z: [batch, latent_dim] latent representation.
			target_ids: [batch, seq_len] ground-truth token ids for teacher forcing.

		Returns:
			logits: [batch, seq_len, vocab_size]
		"""
		h = nnx.tanh(self.z_to_hidden(z))  # [batch, hidden_dim]
		x = self.embedding(target_ids)     # [batch, seq_len, emb_dim]

		# Transpose for jax.lax.scan: [seq_len, batch, emb_dim]
		x = jnp.transpose(x, (1, 0, 2))

		def step(carry, x_t):
			new_carry, y = self.gru_cell(carry, x_t)
			return new_carry, y

		_, hiddens = jax.lax.scan(step, h, x)  # [seq_len, batch, hidden_dim]
		hiddens = jnp.transpose(hiddens, (1, 0, 2))  # [batch, seq_len, hidden_dim]
		return self.output_projection(hiddens)


class PredictionHead(nnx.Module):
	"""MLP regression head: latent vector Z → scalar bioactivity prediction."""

	def __init__(self, latent_dim: int = 128, hidden_dims: Sequence[int] = (256, 128),
				 dropout_rate: float = 0.2, rngs: Optional[nnx.Rngs] = None):
		if rngs is None:
			rngs = nnx.Rngs(0)

		kernel_init = jax.nn.initializers.normal(stddev=0.02)

		self.layers = nnx.List()
		in_dim = latent_dim
		for h_dim in hidden_dims:
			self.layers.append(nnx.Linear(in_features=in_dim, out_features=h_dim, kernel_init=kernel_init, rngs=rngs))
			in_dim = h_dim

		self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
		self.output = nnx.Linear(in_features=in_dim, out_features=1, kernel_init=kernel_init, rngs=rngs)

	def __call__(self, z: jnp.ndarray, is_training: bool = False) -> jnp.ndarray:
		"""Predict scalar bioactivity from latent vector.

		Args:
			z: [batch, latent_dim]
			is_training: enables dropout.

		Returns:
			predictions: [batch]
		"""
		deterministic = not is_training
		x = z
		for layer in self.layers:
			x = layer(x)
			x = nnx.relu(x)
			x = self.dropout(x, deterministic=deterministic)
		return self.output(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Composite models — wire the building blocks for each training stage
# ---------------------------------------------------------------------------

class SmilesAutoencoder(nnx.Module):
	"""Encoder + Decoder. Pretrained on unlabeled SMILES (stage 1)."""

	def __init__(self, encoder: SmilesEncoder, decoder: SmilesDecoder):
		self.encoder = encoder
		self.decoder = decoder

	def __call__(self, input_ids: jnp.ndarray, target_ids: jnp.ndarray, is_training: bool = False):
		"""Returns (logits, z)."""
		z = self.encoder(input_ids, is_training=is_training)
		logits = self.decoder(z, target_ids)
		return logits, z


class EncoderPredictor(nnx.Module):
	"""Encoder + PredictionHead. Pretrained on labeled data (stage 2)."""

	def __init__(self, encoder: SmilesEncoder, prediction_head: PredictionHead):
		self.encoder = encoder
		self.prediction_head = prediction_head

	def __call__(self, input_ids: jnp.ndarray, is_training: bool = False):
		"""Returns (predictions, z)."""
		z = self.encoder(input_ids, is_training=is_training)
		predictions = self.prediction_head(z, is_training=is_training)
		return predictions, z


class JointMolecularModel(nnx.Module):
	"""Encoder + Decoder + PredictionHead. Finetuned jointly (stage 3)."""

	def __init__(self, encoder: SmilesEncoder, decoder: SmilesDecoder, prediction_head: PredictionHead):
		self.encoder = encoder
		self.decoder = decoder
		self.prediction_head = prediction_head

	def __call__(self, input_ids: jnp.ndarray, target_ids: jnp.ndarray, is_training: bool = False):
		"""Returns (logits, predictions, z)."""
		z = self.encoder(input_ids, is_training=is_training)
		logits = self.decoder(z, target_ids)
		predictions = self.prediction_head(z, is_training=is_training)
		return logits, predictions, z
