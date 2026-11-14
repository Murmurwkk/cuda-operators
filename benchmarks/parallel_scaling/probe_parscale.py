"""Probe a ParScale checkpoint: stream diversity and serving latency.

Run with the torch venv:
  .venv-torch/Scripts/python.exe probe_parscale.py --model <dir> --mode probe
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

TEXT = (
    "The history of computing is often told as a sequence of machines: the difference engine, "
    "the tabulator, the stored-program computer, the microprocessor. Each one changed what could "
    "be computed, but the deeper change was in how people described problems. A problem that had "
    "to be phrased as arithmetic became a problem that could be phrased as a procedure, and then "
    "as a data structure. Programming languages followed the same path. Early languages were close "
    "to the machine, with explicit registers and jumps; later ones moved the description upward, "
    "toward collections and constraints and invariants. The compiler took on the burden of "
    "translation, and the programmer could think in larger units. This is not only a story about "
    "abstraction. It is also a story about trust: every layer of abstraction is a promise that the "
    "lower layer will behave, and that promise has to be earned by testing, by specification, and "
    "sometimes by failure. The machines that lasted were not the fastest ones. They were the ones "
    "whose abstractions held under pressure."
)


def load(model_dir, device="cuda"):
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    return tok, model


def stream_stats(model, tok, text, max_tokens=384):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids.cuda()
    captured = {}

    n = model.config.parscale_n
    if n == 1:
        with torch.no_grad():
            logits = model(ids).logits
        targets = ids[:, 1:]
        return {
            "n_streams": 1,
            "tokens": int(ids.shape[1]),
            "agg_logits_self_nll": float(
                torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]), targets.reshape(-1)
                )
            ),
        }

    agg = None
    for name, mod in model.named_modules():
        if name.endswith("aggregate_layer"):
            agg = mod
            break
    if agg is None:
        raise SystemExit("no aggregate_layer found")

    def hook(mod, inp, out):
        captured["x"] = inp[0].detach()
        captured["w"] = out.detach()

    h = agg.register_forward_hook(hook)
    with torch.no_grad():
        logits = model(ids).logits
    h.remove()

    x = captured["x"]  # (P*b, s, h) -> wait: aggregate input is (b, s, P*h)
    w = captured["w"]  # (b, s, P)
    n = model.config.parscale_n
    hidden = model.config.hidden_size
    b, s, ph = x.shape
    streams = x.view(b, s, hidden, n).permute(0, 1, 3, 2).float()  # (b,s,n,h)

    # final weights as used by the model
    ww = torch.softmax(w.float(), dim=-1)
    eps = getattr(model.config, "parscale_attn_smooth", 0.0) or 0.0
    if eps:
        ww = ww * (1 - eps) + eps / n

    # per-stream logits for next-token prediction
    lm = model.lm_head
    per = []
    for i in range(n):
        lg = lm(streams[:, :, i].to(torch.bfloat16)).float()
        per.append(lg)
    per = torch.stack(per, dim=2)  # (b,s,n,V)

    shift_hidden = streams[:, :-1]
    targets = ids[:, 1:]
    nlls = []
    for i in range(n):
        lg = per[:, :-1, i].reshape(-1, per.shape[-1])
        nlls.append(
            torch.nn.functional.cross_entropy(lg, targets.reshape(-1), reduction="none")
            .view(b, s - 1)
            .mean(dim=(0, 1))
        )
    nll = torch.stack(nlls)
    uni = streams.mean(dim=2)
    logits_uni = lm(uni.to(torch.bfloat16)).float()
    nll_uni = float(
        torch.nn.functional.cross_entropy(
            logits_uni[:, :-1].reshape(-1, logits_uni.shape[-1]), targets.reshape(-1)
        )
    )

    pred = per[:, :-1].argmax(-1)  # (b, s-1, n)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((pred[:, :, i] == pred[:, :, j]).float().mean().item())
    agree = {
        "pairwise_top1_agreement": float(statistics.mean(pairs)),
        "pairwise_top1_agreement_min": float(min(pairs)),
        "pairwise_top1_agreement_max": float(max(pairs)),
        "agg_top1_agreement": float((pred[:, :, 0] == logits[:, :-1].argmax(-1)).float().mean()),
    }
    # pairwise cosine between stream hidden states at each position
    hn = torch.nn.functional.normalize(shift_hidden, dim=-1)
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append((hn[:, :, i] * hn[:, :, j]).sum(-1).mean().item())

    wnorm = ww[0].mean(dim=0)  # mean weight per stream over positions
    ent = -(ww.clamp_min(1e-9).log() * ww).sum(-1).mean().item()
    xmax = ww.max(dim=-1).values.mean().item()

    out = {
        "n_streams": n,
        "tokens": int(ids.shape[1]),
        "nll_per_stream": [float(v) for v in nll.tolist()],
        "nll_mean": float(nll.mean()),
        "nll_std": float(nll.std()),
        "nll_uniform_weights": nll_uni,
        "nll_best_stream": float(nll.min()),
        "top1_agreement_vs_mean": agree,
        "pairwise_cosine_mean": float(statistics.mean(sims)),
        "pairwise_cosine_min": float(min(sims)),
        "pairwise_cosine_max": float(max(sims)),
        "weight_mean_per_stream": [float(v) for v in wnorm.tolist()],
        "weight_entropy": ent,
        "weight_max_mean": xmax,
        "agg_logits_self_nll": float(
            torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
        ),
    }
    return out


def latency(model, tok, batch_sizes, prompt_tokens=128, decode_tokens=32):
    # Kept as a compatibility entry point. The old difference-of-minima
    # estimator is retired; all timings now come from explicit cached steps.
    from decode_latency import benchmark
    return benchmark(model, tok, batch_sizes, prompt_tokens, decode_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="probe")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tok, model = load(args.model)
    res = {"model": Path(args.model).name, "params": sum(p.numel() for p in model.parameters())}
    if args.mode in ("probe", "all"):
        res["streams"] = stream_stats(model, tok, TEXT)
    if args.mode in ("latency", "all"):
        res["latency"] = latency(model, tok, [1, 2, 4, 8])
    text = json.dumps(res, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
