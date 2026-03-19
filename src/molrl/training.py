from __future__ import annotations

from typing import Any, Mapping, Optional

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from .models import AutoregressiveTransformer


"""Core training helpers for autoregressive transformer pretraining."""


def _extract_input_ids(batch: Any) -> jnp.ndarray:
	"""Extract input token ids from batch-like objects."""
	if isinstance(batch, Mapping):
		input_ids = batch.get("sequences")
	elif isinstance(batch, (tuple, list)):
		input_ids = batch[0]
	else:
		input_ids = batch

	if input_ids is None:
		raise ValueError("Could not extract input ids from batch")
	return jnp.asarray(input_ids, dtype=jnp.int32)


def _make_attention_mask(input_ids: jnp.ndarray, pad_token_id: Optional[int]) -> Optional[jnp.ndarray]:
	"""Create boolean padding mask, or None when no pad token is provided."""
	if pad_token_id is None:
		return None
	return input_ids != pad_token_id


def transformer_autoregression_loss(model: AutoregressiveTransformer, input_ids: jnp.ndarray, *, pad_token_id: Optional[int] = None, is_training: bool = True) -> jnp.ndarray:
	"""Compute next-token cross-entropy loss for autoregressive transformer training."""
	if input_ids.shape[1] < 2:
		raise ValueError("input_ids must have sequence length >= 2 for autoregressive training")

	attention_mask = _make_attention_mask(input_ids, pad_token_id)
	logits = model(input_ids, attention_mask=attention_mask, is_training=is_training)

	next_token_logits = logits[:, :-1, :]
	next_token_labels = input_ids[:, 1:]

	next_token_targets = jax.nn.one_hot(next_token_labels, num_classes=model.vocab_size, dtype=next_token_logits.dtype)
	per_token_loss = optax.softmax_cross_entropy(next_token_logits, next_token_targets)

	if attention_mask is None:
		valid_targets = jnp.ones_like(next_token_labels, dtype=jnp.float32)
	else:
		valid_targets = attention_mask[:, 1:].astype(jnp.float32)

	token_count = jnp.maximum(valid_targets.sum(), 1.0)
	return (per_token_loss * valid_targets).sum() / token_count


@nnx.jit
def transformer_train_step(model: AutoregressiveTransformer, optimizer: nnx.Optimizer, batch: Any, *, pad_token_id: Optional[int] = None) -> jnp.ndarray:
	"""Run one training step and update model parameters."""
	input_ids = _extract_input_ids(batch)

	def loss_fn(current_model):
		return transformer_autoregression_loss(current_model, input_ids, pad_token_id=pad_token_id, is_training=True)

	loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param))(model)
	optimizer.update(model, grads)
	return loss


@nnx.jit
def transformer_val_step(model: AutoregressiveTransformer, batch: Any, *, pad_token_id: Optional[int] = None) -> jnp.ndarray:
	"""Run one validation step without updating parameters."""
	input_ids = _extract_input_ids(batch)
	return transformer_autoregression_loss(model, input_ids, pad_token_id=pad_token_id, is_training=False)
