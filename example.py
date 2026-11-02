"""Sample from LLaDA-8B-Instruct with a deterministic unmasking rule, then compute the exact
log-likelihood of those samples under the same sampler.

    uv run example.py --rule threshold --mu 0.9 --block_size 32 --gen_len 128 --temperature 0.5
"""

import argparse

import torch
from transformers import AutoModel, AutoTokenizer

import rules
from duel import log_likelihood, make_canvas, sample

MASK_ID = 126336  # LLaDA's [MASK]; its pad id is EOS (126081), taken from the tokenizer

PROMPTS = [
    "What is the capital of France?",
    "Write a short poem about the ocean, then explain in one sentence what it is about.",
]


def build_rule(args) -> rules.Rule:
    rule = {
        "l2r": lambda: rules.LeftToRight(args.k),
        "confidence": lambda: rules.Confidence(args.k),
        "margin": lambda: rules.Margin(args.k),
        "threshold": lambda: rules.ConfidenceThreshold(args.mu),
        "klass": lambda: rules.KLASS(args.mu, args.nu),
    }[args.rule]()
    return rules.Block(rule, args.block_size) if args.block_size > 0 else rule


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--rule", default="threshold", choices=["l2r", "confidence", "margin", "threshold", "klass"])
    p.add_argument("--k", type=int, default=1, help="positions per step (l2r / confidence / margin)")
    p.add_argument("--mu", type=float, default=0.9, help="confidence threshold (threshold / klass)")
    p.add_argument("--nu", type=float, default=0.01, help="KL threshold (klass)")
    p.add_argument("--block_size", type=int, default=32, help="0 = no blocks")
    p.add_argument("--gen_len", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.to("cuda").eval()

    # Prompts of different lengths -> left-padded canvas with gen_len masks per row.
    chats = [tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False)
             for q in PROMPTS]
    prompt_ids = [tok(c, add_special_tokens=False)["input_ids"] for c in chats]
    x, attn = make_canvas(prompt_ids, args.gen_len, MASK_ID, tok.pad_token_id)
    x, attn = x.cuda(), attn.cuda()

    # The denoiser DUEL sees: tokens -> logits. LLaDA's forward pass, with padding masked out.
    denoiser = lambda z: model(z, attention_mask=attn).logits  # noqa: E731
    rule = build_rule(args)

    # Algorithm 1: sample.
    out, nfe = sample(denoiser, rule, x, MASK_ID, temperature=args.temperature)
    gen_mask = x == MASK_ID
    for i, q in enumerate(PROMPTS):
        text = tok.decode(out[i][gen_mask[i]], skip_special_tokens=True)
        print(f"### {q}\n{text.strip()}\n")
    print(f"rule = {rule}\nforward passes = {nfe} for {args.gen_len} tokens per row")

    # Algorithm 2: exact log-likelihood of those samples under the same sampler.
    if args.temperature > 0:
        ll, nfe_ll = log_likelihood(denoiser, rule, out, gen_mask, MASK_ID, temperature=args.temperature)
        for i, v in enumerate(ll.tolist()):
            print(f"row {i}: log p = {v:.2f}  NLL/token = {-v / args.gen_len:.3f}")
        # Scoring retraces the sampling trajectory, so it takes exactly as many forward passes.
        print(f"forward passes when scoring = {nfe_ll} (sampling took {nfe})")


if __name__ == "__main__":
    main()
