#!/usr/bin/env python3
"""
Meta-Router for 32-domain Brainstacks MoE.

Architecture:
    Input: blended hidden state (h_dim=3072)
    -> Linear(3072, 512) + GELU + LayerNorm projection
    -> GlobalAttentionPool (learned query, single-head attention)
    -> DomainCrossAttention (32 learned domain query vectors)
    -> MLPFusion: 512 -> 256 -> 1 per domain
    -> Temperature-scaled sigmoid -> 32 independent scores in [0, 1]

Total: ~1.5M parameters, <5ms inference on Apple Silicon.

All modules use mlx.nn.Module for native Metal acceleration.
"""
from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class GlobalAttentionPool(nn.Module):
    """Single-head attention pooling with a learned query vector.

    Transforms a batch of vectors through key/value projections,
    then attends from a learned query to produce a refined representation.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        # Learned query: (1, dim)
        self.query = mx.random.normal((1, dim)) * 0.02
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.value_proj = nn.Linear(dim, dim, bias=False)
        self.scale = math.sqrt(dim)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (batch, dim) -- single vector per sample (already pooled).
        Returns:
            (batch, dim) -- attention-weighted representation.
        """
        batch_size = x.shape[0]
        # Treat each sample as length-1 sequence
        x_seq = mx.expand_dims(x, axis=1)  # (B, 1, D)
        k = self.key_proj(x_seq)            # (B, 1, D)
        v = self.value_proj(x_seq)          # (B, 1, D)

        # Broadcast query: (1, D) -> (B, 1, D)
        q = mx.broadcast_to(
            mx.expand_dims(self.query, axis=0),
            (batch_size, 1, k.shape[-1]),
        )

        # Attention: (B, 1, D) @ (B, D, 1) -> (B, 1, 1)
        attn_logits = (q @ mx.transpose(k, (0, 2, 1))) / self.scale
        attn_weights = mx.softmax(attn_logits, axis=-1)

        # Weighted value: (B, 1, 1) @ (B, 1, D) -> (B, 1, D)
        out = attn_weights @ v
        return out[:, 0, :]  # (B, D)


class DomainCrossAttention(nn.Module):
    """Cross-attention from 32 learned domain queries to the global representation.

    Each domain query vector attends to the input, producing a per-domain
    feature vector that captures domain-relevant information.
    """

    def __init__(self, dim: int, num_domains: int) -> None:
        super().__init__()
        self.num_domains = num_domains
        self.dim = dim
        # Domain queries: (num_domains, dim)
        self.domain_queries = mx.random.normal((num_domains, dim)) * 0.02
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.value_proj = nn.Linear(dim, dim, bias=False)
        self.scale = math.sqrt(dim)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (batch, dim) -- global representation from attention pool.
        Returns:
            (batch, num_domains, dim) -- per-domain attended features.
        """
        batch_size = x.shape[0]
        # Key/value from input: (B, 1, D)
        k = mx.expand_dims(self.key_proj(x), axis=1)
        v = mx.expand_dims(self.value_proj(x), axis=1)

        # Domain queries: (num_domains, D) -> (B, num_domains, D)
        q = mx.broadcast_to(
            mx.expand_dims(self.domain_queries, axis=0),
            (batch_size, self.num_domains, self.dim),
        )

        # Attention: (B, num_domains, D) @ (B, D, 1) -> (B, num_domains, 1)
        attn_logits = (q @ mx.transpose(k, (0, 2, 1))) / self.scale
        attn_weights = mx.softmax(attn_logits, axis=-1)

        # Weighted value: (B, num_domains, 1) @ (B, 1, D) -> (B, num_domains, D)
        out = attn_weights @ v
        return out


class MLPFusion(nn.Module):
    """Per-domain MLP that fuses cross-attention output to a scalar gate.

    Maps each domain's feature vector through a 2-layer MLP:
    dim -> dim//2 (GELU + dropout) -> 1 (raw logit).
    """

    def __init__(self, dim: int, num_domains: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // 2)
        self.fc2 = nn.Linear(dim // 2, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.num_domains = num_domains

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (batch, num_domains, dim) -- per-domain features.
        Returns:
            (batch, num_domains) -- raw logits (pre-sigmoid).
        """
        h = self.fc1(x)       # (B, num_domains, dim//2)
        h = nn.gelu(h)
        h = self.dropout(h)
        h = self.fc2(h)       # (B, num_domains, 1)
        return h[..., 0]      # (B, num_domains)


class MetaRouter(nn.Module):
    """
    32-sigmoid meta-router for Brainstacks domain selection.

    Takes a blended hidden state vector (mid + last layer) and outputs
    32 independent sigmoid scores indicating which stacks to activate.

    Pipeline:
        input_proj (Linear + GELU + LayerNorm)
        -> global_attn (learned query attention pool) + residual
        -> domain_cross_attn (32 domain query vectors)
        -> mlp_fusion (per-domain MLP -> scalar)
        -> temperature-scaled sigmoid

    Args:
        h_dim: Hidden dimension of the base model (3072 for Qwen3.5-4B).
        hidden_dim: Internal projection dimension (512).
        num_domains: Number of domain stacks (32).
        dropout: Dropout rate for MLP fusion (0.1).
        temperature_init: Initial temperature for sigmoid scaling (2.0).
    """

    def __init__(
        self,
        h_dim: int = 3072,
        hidden_dim: int = 512,
        num_domains: int = 32,
        dropout: float = 0.1,
        temperature_init: float = 2.0,
    ) -> None:
        super().__init__()
        self.h_dim = h_dim
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains

        # Stage 1: Project from model hidden dim to router hidden dim
        self.proj_linear = nn.Linear(h_dim, hidden_dim)
        self.proj_norm = nn.LayerNorm(hidden_dim)

        # Stage 2: Global attention pooling
        self.global_attn = GlobalAttentionPool(hidden_dim)

        # Stage 3: Domain cross-attention
        self.domain_cross_attn = DomainCrossAttention(hidden_dim, num_domains)

        # Stage 4: MLP fusion -> logits
        self.mlp_fusion = MLPFusion(hidden_dim, num_domains, dropout)

        # Learnable temperature (stored as log for positivity via softplus)
        self._log_temperature = mx.array([math.log(temperature_init)])

    def input_proj(self, x: mx.array) -> mx.array:
        """Project input from h_dim to hidden_dim with GELU + LayerNorm."""
        h = self.proj_linear(x)
        h = nn.gelu(h)
        h = self.proj_norm(h)
        return h

    def get_temperature(self) -> mx.array:
        """Return the positive temperature value."""
        return mx.exp(self._log_temperature)

    def __call__(self, hidden: mx.array) -> mx.array:
        """
        Args:
            hidden: (batch, h_dim) -- blended mid+last hidden state.
        Returns:
            (batch, num_domains) -- sigmoid scores in [0, 1].
        """
        # Project to router dim
        projected = self.input_proj(hidden)  # (B, hidden_dim)

        # Global attention pool
        pooled = self.global_attn(projected)  # (B, hidden_dim)

        # Residual connection
        pooled = pooled + projected

        # Domain cross-attention
        domain_features = self.domain_cross_attn(pooled)  # (B, num_domains, hidden_dim)

        # MLP fusion -> logits
        logits = self.mlp_fusion(domain_features)  # (B, num_domains)

        # Temperature-scaled sigmoid
        temp = self.get_temperature()
        scores = mx.sigmoid(logits / temp)

        return scores

    def get_active_stacks(
        self,
        scores: mx.array,
        gate_threshold: float = 0.12,
        chat_floor: float = 0.20,
        max_active: int = 4,
    ) -> list[list[tuple[int, float]]]:
        """
        Apply inference rules to router scores.

        Rules:
        1. chat-fr (domain 0) score is raised to at least chat_floor.
        2. Only stacks with effective score >= gate_threshold are considered.
        3. At most max_active stacks are returned, sorted by score descending.

        Args:
            scores: (batch, num_domains) sigmoid outputs.
            gate_threshold: Minimum score to consider a stack.
            chat_floor: Minimum score for chat-fr (domain 0).
            max_active: Maximum simultaneous stacks.

        Returns:
            List of lists (one per batch item), each containing
            (domain_index, score) tuples sorted by score descending.
        """
        # Convert to numpy for iteration
        scores_np = scores.tolist()
        if not isinstance(scores_np[0], list):
            scores_np = [scores_np]

        batch_results: list[list[tuple[int, float]]] = []

        for sample_scores in scores_np:
            # Apply chat floor
            effective = list(sample_scores)
            effective[0] = max(effective[0], chat_floor)

            # Filter by gate threshold
            active: list[tuple[int, float]] = []
            for domain_idx in range(self.num_domains):
                score = effective[domain_idx]
                if score >= gate_threshold:
                    active.append((domain_idx, score))

            # Sort by score descending, take top max_active
            active.sort(key=lambda x: x[1], reverse=True)
            active = active[:max_active]

            batch_results.append(active)

        return batch_results
