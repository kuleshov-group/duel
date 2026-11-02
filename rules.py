"""Deterministic unmasking rules (Table 1 of the DUEL paper).

A rule is called once per denoising step:

    reveal = rule(probs, masked)

    probs  : [B, L, V] float  denoiser's predicted distribution at every position
    masked : [B, L]    bool   positions that are still masked (prompt / padding are never masked)
    reveal : [B, L]    bool   positions to reveal this step

Contract (what makes DUEL exact and the sampling loop terminate):
  * reveal is a subset of masked,
  * every row that still has masked positions reveals at least one of them,
  * the output is a deterministic function of (probs, masked) -- plus, for stateful rules
    such as KLASS, of earlier steps of the same trajectory. Call `reset()` before each new batch.

Rules never see prompts or padding: those positions are simply never in `masked`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


# ----------------------------------------------------------------------------- helpers


def _topk(scores: Tensor, masked: Tensor, k: int) -> Tensor:
    """The (up to) k highest-scoring masked positions in each row."""
    scores = scores.masked_fill(~masked, float("-inf"))
    idx = scores.topk(min(k, scores.shape[-1]), dim=-1).indices
    reveal = torch.zeros_like(masked).scatter_(-1, idx, True)
    return reveal & masked  # drop picks that landed on unmasked positions (rows with < k masks)


def _threshold(scores: Tensor, masked: Tensor, mu: float) -> Tensor:
    """All masked positions with score >= mu; falls back to the single best one so each row progresses."""
    reveal = masked & (scores >= mu)
    fallback = _topk(scores, masked, 1) & ~reveal.any(-1, keepdim=True)
    return reveal | fallback


def _confidence(probs: Tensor) -> Tensor:
    return probs.max(dim=-1).values  # p^(1)


class Rule:
    def reset(self) -> None:
        """Clear per-trajectory state. Called by `sample` / `log_likelihood` at the start."""

    def __call__(self, probs: Tensor, masked: Tensor) -> Tensor:
        raise NotImplementedError


# ----------------------------------------------------------------------------- rules


@dataclass
class LeftToRight(Rule):
    """The k leftmost masked positions. k=1 is autoregressive order."""

    k: int = 1

    def __call__(self, probs, masked):
        pos = torch.arange(masked.shape[-1], device=masked.device, dtype=probs.dtype)
        return _topk(-pos.expand(masked.shape), masked, self.k)


@dataclass
class Confidence(Rule):
    """Greedy confidence (Nie et al., 2025): the k masked positions with the largest p^(1)."""

    k: int = 1

    def __call__(self, probs, masked):
        return _topk(_confidence(probs), masked, self.k)


@dataclass
class Margin(Rule):
    """Probability margin (Kim et al., 2025): the k masked positions with the largest p^(1) - p^(2)."""

    k: int = 1

    def __call__(self, probs, masked):
        top2 = probs.topk(2, dim=-1).values
        return _topk(top2[..., 0] - top2[..., 1], masked, self.k)


@dataclass
class ConfidenceThreshold(Rule):
    """Confidence threshold (Fast-dLLM, Wu et al., 2025): every masked position with p^(1) >= mu.

    Adaptive: the number of positions per step (and hence of steps) is not known in advance.
    """

    mu: float = 0.9

    def __call__(self, probs, masked):
        return _threshold(_confidence(probs), masked, self.mu)


@dataclass
class KLASS(Rule):
    """KLASS (Kim et al., 2025): p^(1) >= mu AND KL(previous prediction || current prediction) <= nu.

    Stateful: it remembers the previous step's probs. On the first step the KL term is 0, so it
    coincides with ConfidenceThreshold (paper, App. C). Falls back to the most confident position.
    """

    mu: float = 0.9
    nu: float = 0.01
    _prev: Tensor | None = field(default=None, init=False, repr=False)

    def reset(self):
        self._prev = None

    def __call__(self, probs, masked):
        conf = _confidence(probs)
        if self._prev is None:
            kl = torch.zeros_like(conf)
        else:
            eps = 1e-30
            kl = (self._prev * (self._prev.clamp_min(eps).log() - probs.clamp_min(eps).log())).sum(-1)
        self._prev = probs
        reveal = masked & (conf >= self.mu) & (kl <= self.nu)
        fallback = _topk(conf, masked, 1) & ~reveal.any(-1, keepdim=True)
        return reveal | fallback


@dataclass
class Block(Rule):
    """Compose any rule with block decoding (Arriola et al., 2025).

    The generation region of each row is cut into blocks of `block_size`, counted from the row's
    first masked position. The inner rule only sees the leftmost block that still has masks, so
    blocks are completed left to right. Each row tracks its own block, so rows in a batch may be
    in different blocks (e.g. left-padded continuations of different lengths).
    """

    rule: Rule
    block_size: int
    _start: Tensor | None = field(default=None, init=False, repr=False)

    def reset(self):
        self._start = None
        self.rule.reset()

    def __call__(self, probs, masked):
        if self._start is None:  # first call: masked == the whole generation region
            self._start = masked.int().argmax(-1)  # first masked column per row
        pos = torch.arange(masked.shape[-1], device=masked.device)
        block = (pos - self._start[:, None]).div(self.block_size, rounding_mode="floor")  # [B, L]
        current = block.masked_fill(~masked, torch.iinfo(block.dtype).max).min(-1, keepdim=True).values
        return self.rule(probs, masked & (block == current))
