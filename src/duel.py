"""DUEL: sampling (Algorithm 1) and exact log-likelihood (Algorithm 2) for masked diffusion LMs.

Both algorithms run the same loop -- same rule, same revealed sets, same forward passes. They
differ in one line: sampling draws a token at each revealed position, scoring adds the
log-probability of the true token there.

The denoiser is any callable  x: Long[B, L] -> logits: [B, L, V]  where logits[:, l] is the
prediction for position l. For LLaDA this is `lambda x: model(x, attention_mask=attn).logits`
(see example.py).

Batches are left-padded and described by two tensors:
    x               Long[B, L]   tokens; `mask_id` marks positions to generate / score
    attention_mask  Bool[B, L]   True on real tokens, False on padding (only the denoiser uses it)
`make_canvas` and `make_scoring_batch` build them from lists of token ids.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from rules import Rule

Denoiser = Callable[[Tensor], Tensor]


def _predict(denoiser: Denoiser, x: Tensor, mask_id: int) -> Tensor:
    """One forward pass -> logits [B, L, V] (float32, [MASK] itself never predicted)."""
    logits = denoiser(x).float()
    logits[..., mask_id] = float("-inf")
    return logits


def _check(reveal: Tensor, masked: Tensor) -> None:
    assert not (reveal & ~masked).any(), "rule revealed an already-revealed position"
    assert (reveal.any(-1) | ~masked.any(-1)).all(), "rule revealed nothing for a row with masks left"


@torch.no_grad()
def sample(denoiser: Denoiser, rule: Rule, x: Tensor, mask_id: int, temperature: float = 1.0):
    """Algorithm 1. Fill every `mask_id` position of x. Returns (x, number of forward passes).

    Rules always see the untempered softmax(logits), so e.g. a confidence threshold means the same at
    every temperature; tokens are drawn from softmax(logits / temperature) (argmax if temperature=0).
    """
    x = x.clone()
    masked = x == mask_id
    rule.reset()
    nfe = 0
    while masked.any():  # adaptive rules: the number of steps is not known in advance
        logits = _predict(denoiser, x, mask_id)  # one forward pass
        nfe += 1
        reveal = rule(logits.softmax(-1), masked)  # positions to reveal
        _check(reveal, masked)
        logits = logits[reveal]  # [num_revealed, V]
        if temperature > 0:
            tokens = torch.multinomial((logits / temperature).softmax(-1), 1).squeeze(-1)  # x^l ~ Cat(P_l)
        else:
            tokens = logits.argmax(-1)
        x[reveal] = tokens
        masked &= ~reveal
    return x, nfe


@torch.no_grad()
def log_likelihood(
    denoiser: Denoiser, rule: Rule, x: Tensor, gen_mask: Tensor, mask_id: int, temperature: float = 1.0
):
    """Algorithm 2. Exact log p(x[gen_mask] | rest of x) under the sampler (denoiser, rule, temperature).

    x        : Long[B, L] complete sequences (prompt + continuation, left-padded)
    gen_mask : Bool[B, L] positions to score (the continuation); everything else stays revealed
    Returns (log-likelihood [B], number of forward passes).
    As in `sample`, the rule sees softmax(logits) and tokens are scored under softmax(logits / temperature).
    """
    assert temperature > 0, "the greedy (temperature=0) sampler is a point mass; score at temperature > 0"
    z = x.masked_fill(gen_mask, mask_id)
    masked = gen_mask.clone()
    rule.reset()
    ll = torch.zeros(x.shape[0], device=x.device)
    nfe = 0
    while masked.any():
        logits = _predict(denoiser, z, mask_id)  # one forward pass
        nfe += 1
        reveal = rule(logits.softmax(-1), masked)  # same rule, same order
        logp = (logits / temperature).log_softmax(-1)
        _check(reveal, masked)
        logp_true = logp.gather(-1, x[..., None]).squeeze(-1)  # log P_l[x^l]
        ll += torch.where(reveal, logp_true, 0.0).sum(-1)  # score the truth
        z = torch.where(reveal, x, z)
        masked &= ~reveal
    return ll, nfe


# ----------------------------------------------------------------------------- batching helpers


def make_canvas(prompts: list[list[int]], gen_len: int, mask_id: int, pad_id: int):
    """Left-pad prompts and append `gen_len` masks. Returns (x, attention_mask), both [B, L]."""
    L = max(map(len, prompts)) + gen_len
    x = torch.full((len(prompts), L), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(len(prompts), L, dtype=torch.bool)
    for i, p in enumerate(prompts):
        start = L - gen_len - len(p)
        x[i, start : L - gen_len] = torch.tensor(p)
        attention_mask[i, start:] = True
    x[:, L - gen_len :] = mask_id
    return x, attention_mask


def make_scoring_batch(prompts: list[list[int]], continuations: list[list[int]], pad_id: int):
    """Left-pad prompt + continuation pairs. Returns (x, attention_mask, gen_mask), all [B, L];
    gen_mask marks the continuation tokens to score."""
    L = max(len(p) + len(c) for p, c in zip(prompts, continuations))
    x = torch.full((len(prompts), L), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(len(prompts), L, dtype=torch.bool)
    gen_mask = torch.zeros(len(prompts), L, dtype=torch.bool)
    for i, (p, c) in enumerate(zip(prompts, continuations)):
        x[i, L - len(p) - len(c) :] = torch.tensor(p + c)
        attention_mask[i, L - len(p) - len(c) :] = True
        gen_mask[i, L - len(c) :] = True
    return x, attention_mask, gen_mask
