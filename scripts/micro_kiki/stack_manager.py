#!/usr/bin/env python3
"""
Disk-offloaded stack management with LRU cache.

Stacks live on disk as safetensors files. The StackManager loads them
into memory on demand and evicts the least-recently-used stack when the
cache is full. This keeps memory usage bounded to at most cache_size
stacks simultaneously.

Directory structure:
    output/micro-kiki/stacks/
    +-- chat-fr/
    |   +-- adapters.safetensors
    +-- reasoning/
    |   +-- adapters.safetensors
    +-- ...

Each stack directory contains adapters.safetensors with the MoE-LoRA
weights for that domain, saved by train_stack.py.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger(__name__)

# Domain names in curriculum order (must match brainstacks.yaml)
DOMAIN_NAMES: list[str] = [
    "chat-fr", "reasoning", "python", "typescript", "cpp",
    "rust", "html-css", "shell", "sql", "yaml-json",
    "docker", "kicad-dsl", "spice", "lua-upy", "embedded",
    "stm32", "iot", "freecad", "platformio", "power",
    "emc", "dsp", "spice-sim", "electronics", "kicad-pcb",
    "web-frontend", "web-backend", "music-audio", "devops", "llm-orch",
    "math", "security",
]

NUM_DOMAINS: int = 32


class StackManager:
    """
    LRU-cached disk-offloaded stack loader.

    Maintains at most `cache_size` stacks in memory simultaneously.
    When the cache is full and a new stack is requested, the least
    recently used stack is evicted.

    Supports both safetensors and .npz weight files.
    """

    def __init__(
        self,
        stacks_dir: str | Path,
        cache_size: int = 6,
        device: str | None = None,
    ) -> None:
        self.stacks_dir = Path(stacks_dir)
        self.cache_size = cache_size

        # OrderedDict acts as LRU: most recently accessed at end
        self._cache: OrderedDict[int, dict[str, mx.array]] = OrderedDict()

        # Discover available stacks on disk
        self._stack_paths: dict[int, Path] = {}
        self._discover_stacks()

    def _discover_stacks(self) -> None:
        """Scan stacks_dir for available domain adapters."""
        for idx, name in enumerate(DOMAIN_NAMES):
            domain_dir = self.stacks_dir / name
            if domain_dir.is_dir():
                # Priority order for adapter files
                for filename in [
                    "adapters.safetensors",
                    "adapter.safetensors",
                    "adapter.pt",
                    "adapter.npz",
                ]:
                    adapter_path = domain_dir / filename
                    if adapter_path.exists():
                        self._stack_paths[idx] = adapter_path
                        break
        logger.info(
            "Discovered %d/%d stacks in %s",
            len(self._stack_paths),
            NUM_DOMAINS,
            self.stacks_dir,
        )

    @property
    def num_available_stacks(self) -> int:
        """Number of stacks found on disk."""
        return len(self._stack_paths)

    @property
    def num_loaded(self) -> int:
        """Number of stacks currently in the cache."""
        return len(self._cache)

    def is_loaded(self, domain_idx: int) -> bool:
        """Check if a stack is currently in the cache."""
        return domain_idx in self._cache

    def load(self, domain_idx: int) -> dict[str, mx.array]:
        """
        Load a stack's weights, using cache if available.

        If the stack is already cached, moves it to the end (most recent).
        If not cached and cache is full, evicts the LRU entry first.

        Args:
            domain_idx: Domain index (0-31).

        Returns:
            Dict of weight name -> mx.array.

        Raises:
            IndexError: If domain_idx is out of range or not available.
        """
        if domain_idx < 0 or domain_idx >= NUM_DOMAINS:
            raise IndexError(
                f"Domain index {domain_idx} out of range [0, {NUM_DOMAINS})"
            )
        if domain_idx not in self._stack_paths:
            raise IndexError(
                f"Stack for domain {domain_idx} ({DOMAIN_NAMES[domain_idx]}) "
                f"not found in {self.stacks_dir}"
            )

        # Cache hit: move to end (most recently used)
        if domain_idx in self._cache:
            self._cache.move_to_end(domain_idx)
            return self._cache[domain_idx]

        # Cache miss: evict LRU if needed
        while len(self._cache) >= self.cache_size:
            evicted_idx, evicted_weights = self._cache.popitem(last=False)
            del evicted_weights
            logger.debug(
                "Evicted stack %d (%s) from cache",
                evicted_idx,
                DOMAIN_NAMES[evicted_idx],
            )

        # Load from disk
        path = self._stack_paths[domain_idx]
        weights = self._load_weights_from_disk(path)

        self._cache[domain_idx] = weights
        logger.debug(
            "Loaded stack %d (%s) from %s",
            domain_idx,
            DOMAIN_NAMES[domain_idx],
            path,
        )
        return weights

    def _load_weights_from_disk(self, path: Path) -> dict[str, mx.array]:
        """Load weights from a file on disk, handling multiple formats."""
        if path.suffix == ".safetensors":
            from safetensors.mlx import load_file
            return load_file(str(path))
        elif path.suffix == ".npz":
            return dict(mx.load(str(path)))
        elif path.suffix == ".pt":
            # PyTorch .pt files loaded via torch then converted
            import torch
            data = torch.load(str(path), map_location="cpu", weights_only=True)
            return {k: mx.array(v.numpy()) for k, v in data.items()}
        else:
            raise ValueError(f"Unsupported weight format: {path.suffix}")

    def unload(self, domain_idx: int) -> None:
        """Explicitly remove a stack from the cache."""
        if domain_idx in self._cache:
            del self._cache[domain_idx]

    def clear(self) -> None:
        """Remove all stacks from the cache."""
        self._cache.clear()

    def load_active_stacks(
        self,
        active_list: list[tuple[int, float]],
    ) -> dict[int, dict[str, mx.array]]:
        """
        Load all stacks from an active list (from router output).

        Args:
            active_list: List of (domain_idx, score) tuples from the router.

        Returns:
            Dict of domain_idx -> weight dict for all active stacks.
        """
        result: dict[int, dict[str, mx.array]] = {}
        for domain_idx, _score in active_list:
            try:
                result[domain_idx] = self.load(domain_idx)
            except IndexError:
                logger.warning(
                    "Stack %d (%s) not available on disk, skipping",
                    domain_idx,
                    DOMAIN_NAMES[domain_idx] if domain_idx < NUM_DOMAINS else "?",
                )
        return result
