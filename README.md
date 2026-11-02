# DUEL: Exact Likelihood Reshapes Masked Diffusion Scaling Laws and Beyond

[Gilad Turok](https://giladturok.github.io), [Yair Schiff](https://yair-schiff.github.io), [Chris De Sa](https://www.cs.cornell.edu/~cdesa/)\*, [Volodymyr Kuleshov](https://www.cs.cornell.edu/~kuleshov/)\*
<br>Cornell University &nbsp;&middot;&nbsp; \*Joint senior authorship

[![arXiv](https://img.shields.io/badge/arXiv-2603.01367-b31b1b)](https://arxiv.org/abs/2603.01367)
[![Website](https://img.shields.io/badge/Project-Website-blue)](https://giladturok.github.io/duel/)

> **Note.** The full code (experiments, scaling laws, evaluations) is coming soon. This repository is a
> minimal reference implementation of DUEL to get started: two short files, `duel.py` and `rules.py`.

A masked diffusion model (MDM) sampler is a pretrained denoiser plus an unmasking rule that decides which
positions to reveal at each step. Frontier MDMs use *deterministic* rules such as confidence, so the sampler
follows a single unmasking order per sequence and its likelihood is *exact* rather than a sum over all orders.
DUEL computes it with the same loop as sampling: replay generation, reveal the true tokens instead of sampled
ones, and sum their log-probabilities. No retraining, any pretrained MDM.

## The two algorithms

Sampling and scoring run the same loop (same rule, same revealed sets, same forward passes) and differ in
one line:

```
sample(denoiser, rule)                     log_likelihood(x, denoiser, rule)
  S = {}                                     S = {}; ll = 0
  while masked tokens remain:                while masked tokens remain:
    P = denoiser(x revealed on S)              P = denoiser(x revealed on S)
    I = rule(P, masked)                        I = rule(P, masked)
    for l in I: x[l] ~ Cat(P[l])               for l in I: ll += log P[l][x[l]]
    S = S | I                                  S = S | I
  return x                                   return ll   # exact log-likelihood
```

These are `sample` and `log_likelihood` in [`duel.py`](duel.py).

## Quickstart

```bash
uv sync                      # torch (CUDA 12.8 wheels) + transformers==4.49.0
uv run example.py --rule threshold --block_size 32 --gen_len 128 --temperature 0.5
```

Without uv: `pip install torch transformers==4.49.0`. `example.py` runs LLaDA-8B-Instruct; flags are
`--model`, `--rule {l2r,confidence,margin,threshold,klass}`, `--k`, `--mu`, `--nu`,
`--block_size` (0 = no blocks), `--gen_len`, `--temperature`.

Using the library directly:

```python
import torch
from transformers import AutoModel, AutoTokenizer
from duel import make_canvas, make_scoring_batch, sample, log_likelihood
from rules import Block, ConfidenceThreshold

name = "GSAI-ML/LLaDA-8B-Instruct"
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
model = AutoModel.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
MASK, PAD = 126336, 126081
rule = Block(ConfidenceThreshold(mu=0.9), block_size=32)

prompts = [tok("The capital of France is")["input_ids"], tok("2 + 2 =")["input_ids"]]
x, attn = (t.cuda() for t in make_canvas(prompts, gen_len=64, mask_id=MASK, pad_id=PAD))
out, nfe = sample(lambda z: model(z, attention_mask=attn).logits, rule, x, MASK, temperature=0.5)

conts = [[int(t) for t in row[-64:]] for row in out]  # score the generations under the same sampler
x, attn, gen = (t.cuda() for t in make_scoring_batch(prompts, conts, pad_id=PAD))
ll, nfe = log_likelihood(lambda z: model(z, attention_mask=attn).logits, rule, x, gen, MASK, temperature=0.5)
```

Rules always see the untempered prediction `softmax(logits)`, so a threshold such as `mu=0.9` means the same at
every temperature; `temperature` only changes the distribution tokens are drawn from (and scored under).
`log_likelihood` requires `temperature > 0`: the greedy sampler is a point mass.

## Unmasking rules

All rules live in [`rules.py`](rules.py). $p^{(1)}, p^{(2)}$ are the largest and second-largest predicted
probabilities at a position.

| Class | Hyperparameters | Tokens per step | Reveals | Reference |
|---|---|---|---|---|
| `LeftToRight` | `k` | fixed `k` | `k` leftmost masked positions (`k=1`: AR order) | — |
| `Confidence` | `k` | fixed `k` | top-`k` by $p^{(1)}$ | Nie et al., 2025 |
| `Margin` | `k` | fixed `k` | top-`k` by $p^{(1)} - p^{(2)}$ | Kim et al., 2025 |
| `ConfidenceThreshold` | `mu` | adaptive | all with $p^{(1)} \ge \mu$ | Wu et al., 2025 (Fast-dLLM) |
| `KLASS` | `mu`, `nu` | adaptive | $p^{(1)} \ge \mu$ and KL(prev ‖ current) $\le \nu$ | Kim et al., 2025 (KLASS) |
| `Block(rule, block_size)` | `block_size` | inner rule's | inner rule, restricted to the leftmost unfinished block | Arriola et al., 2025 |

Adaptive rules fall back to the single most confident position when nothing passes the threshold, so every
step makes progress; the number of steps is not known in advance.

## Batching, prompts and padding

A batch is a left-padded `x: Long[B, L]` in which `mask_id` marks the positions to generate (sampling) or
score (`gen_mask`, scoring), plus an explicit boolean `attention_mask`. The attention mask is not derived
from the pad id because LLaDA's pad id is also its EOS token, which legitimately appears in generations.
`make_canvas` and `make_scoring_batch` build both from lists of token ids. Rules only ever see masked
positions, so prompts and padding are never revealed. Rows can finish at different steps; finished rows are
carried along until the whole batch is done. `Block` tracks each row's own block start, so rows whose
generation regions begin at different columns stay correct.

## Writing your own rule

A rule is called once per step as `reveal = rule(probs, masked)` with `probs: [B, L, V]` and
`masked, reveal: Bool[B, L]`. Subclass `rules.Rule` and guarantee that:

- `reveal` is a subset of `masked`;
- every row that still has masked positions reveals at least one;
- the output is a deterministic function of `(probs, masked)` (and, for stateful rules, of earlier steps of
  the same trajectory; clear that state in `reset()`).

Rules must not depend on the sampled tokens. LLaDA's stock `generate` ranks positions by the probability of
the token it just sampled, which is random at temperature > 0; DUEL rules use only the predicted
distribution, and the two coincide at temperature 0.

Other masked diffusion models need their own small denoiser wrapper returning `[B, L, V]` logits where
position `l` predicts token `l` (e.g. for models whose logits are shifted by one position).

## Citation

```bibtex
@article{turok2026duel,
  title   = {{DUEL}: Exact Likelihood Reshapes Masked Diffusion Scaling Laws and Beyond},
  author  = {Turok, Gilad and Schiff, Yair and De Sa, Chris and Kuleshov, Volodymyr},
  journal = {arXiv preprint arXiv:2603.01367},
  year    = {2026}
}
```
