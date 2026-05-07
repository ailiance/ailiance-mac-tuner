#!/usr/bin/env python3
"""
Extract and blend mid-layer + last-layer hidden states from Qwen3.5-4B.

The meta-router needs a single vector per prompt that captures both
intermediate reasoning (mid-layer) and final representation (last-layer).
The blending ratio 0.45/0.55 is from the Brainstacks spec.

Uses the frozen base model loaded via mlx_lm.load. The model is run
with output_hidden_states=True to capture all layer representations,
then mid and last are blended and pooled to a single (B, 3072) vector.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


# Default blending weights from plan spec
MID_LAYER_WEIGHT: float = 0.45
LAST_LAYER_WEIGHT: float = 0.55


def get_mid_layer_index(model) -> int:
    """Return the index of the middle hidden layer.

    Works with both mlx_lm models (model.model.layers) and mock models
    that expose config.num_hidden_layers.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
    elif hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    elif hasattr(model, "num_layers"):
        num_layers = model.num_layers
    else:
        raise AttributeError(
            "Cannot determine num_layers from model. "
            "Expected model.model.layers, model.config.num_hidden_layers, "
            "or model.num_layers."
        )
    return num_layers // 2


def pool_last_token(hidden_states: mx.array) -> mx.array:
    """Pool by taking the last token's hidden state.

    Args:
        hidden_states: (batch, seq_len, h_dim)
    Returns:
        (batch, h_dim)
    """
    return hidden_states[:, -1, :]


def pool_mean(hidden_states: mx.array) -> mx.array:
    """Pool by averaging across the sequence dimension.

    Args:
        hidden_states: (batch, seq_len, h_dim)
    Returns:
        (batch, h_dim)
    """
    return mx.mean(hidden_states, axis=1)


def _run_with_hidden_states(model, input_ids: mx.array) -> list[mx.array]:
    """Run a forward pass through the Qwen model collecting hidden states.

    For mlx_lm Qwen models, we manually walk the layers to capture
    intermediate hidden states, since the standard __call__ only
    returns final logits.

    Args:
        model: The mlx_lm Qwen model.
        input_ids: (batch, seq_len) token IDs.

    Returns:
        List of hidden states, one per layer plus the embedding output.
        Index 0 = embedding output, index i = output of layer i-1,
        index -1 = final layer output (pre-head).
    """
    # For mock models that support output_hidden_states directly
    if hasattr(model, "forward") and not hasattr(model, "model"):
        outputs = model(input_ids, output_hidden_states=True)
        return list(outputs.hidden_states)

    # For mlx_lm Qwen models: walk through the transformer layers
    inner = model.model if hasattr(model, "model") else model

    # Embedding
    h = inner.embed_tokens(input_ids)
    hidden_states = [h]

    # Transformer layers
    mask = nn.MultiHeadAttention.create_additive_causal_mask(
        h.shape[1]
    ).astype(h.dtype)

    for layer in inner.layers:
        h = layer(h, mask=mask)
        hidden_states.append(h)

    # Final layer norm
    if hasattr(inner, "norm"):
        h = inner.norm(h)
        hidden_states[-1] = h

    return hidden_states


def extract_blended_hidden(
    model,
    input_ids: mx.array,
    mid_weight: float = MID_LAYER_WEIGHT,
    last_weight: float = LAST_LAYER_WEIGHT,
    pool_fn: str = "last_token",
) -> mx.array:
    """
    Run a forward pass and blend mid-layer + last-layer hidden states.

    The mid-layer captures intermediate reasoning patterns while the
    last layer captures the final semantic representation. Blending
    both gives the router a richer signal for domain classification.

    Args:
        model: The base language model (Qwen3.5-4B via mlx_lm).
        input_ids: (batch, seq_len) token IDs.
        mid_weight: Weight for mid-layer hidden state (default 0.45).
        last_weight: Weight for last-layer hidden state (default 0.55).
        pool_fn: Pooling strategy -- "last_token" or "mean".

    Returns:
        (batch, h_dim) blended hidden state vector.

    Raises:
        ValueError: If weights don't sum to 1.0.
    """
    if abs(mid_weight + last_weight - 1.0) > 1e-4:
        raise ValueError(
            f"mid_weight ({mid_weight}) + last_weight ({last_weight}) "
            f"must sum to 1.0, got {mid_weight + last_weight:.4f}"
        )

    pooler = pool_last_token if pool_fn == "last_token" else pool_mean

    hidden_states = _run_with_hidden_states(model, input_ids)

    mid_idx = get_mid_layer_index(model)
    mid_hidden = pooler(hidden_states[mid_idx])   # (B, h_dim)
    last_hidden = pooler(hidden_states[-1])        # (B, h_dim)

    blended = mid_weight * mid_hidden + last_weight * last_hidden
    mx.eval(blended)
    return blended
