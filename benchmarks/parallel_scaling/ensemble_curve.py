"""How much of the P=8 gain survives when streams are averaged.

Loads one ParScale checkpoint, recovers per-stream logits from the
aggregator input, then evaluates the NLL of averaging every k-subset of
streams (k=1..8), the per-token oracle, and the learned gate.

Every NLL is a token-weighted mean over all predicted positions of the text.
The statistics for each k (mean, min, max) are taken across the C(P, k)
subsets, so for k = P there is exactly one value. The chunking below only
limits memory; it does not change any result.

Correction (2026-11-13): the first version appended one value per (chunk,
subset) pair, so its min/max mixed chunk-to-chunk variation with variation
between subsets, and its mean weighted the short final chunk like a full one.
"""

import argparse
import itertools
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from probe_parscale import TEXT, load
from evidence import provenance, sha256
import hashlib


def subset_nll(logits, targets, comb):
    """NLL of the uniform average of logits over the streams in comb."""
    lp = logits[:, comb, :].mean(dim=1).float().log_softmax(-1)
    return float(-lp.gather(1, targets[:, None]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tok, model = load(args.model, args.device)
    ids = tok(TEXT, return_tensors="pt").input_ids.to(args.device)
    P = model.config.parscale_n
    hidden = model.config.hidden_size
    if P == 1:
        raise SystemExit("this analysis needs P>1")

    agg = None
    for name, mod in model.named_modules():
        if name.endswith("aggregate_layer"):
            agg = mod
            break
    captured = {}

    def hook(mod, inp, out):
        captured["x"] = inp[0].detach()
        captured["w"] = out.detach()

    h = agg.register_forward_hook(hook)
    with torch.no_grad():
        model(ids)
    h.remove()

    # aggregate input is (b, s, P*h) with hidden-major, stream-minor order
    x = captured["x"][0]                      # (s, P*h)
    w = captured["w"][0].float()              # (s, P), pre-softmax
    w = torch.softmax(w, dim=-1)
    eps = getattr(model.config, "parscale_attn_smooth", 0.0) or 0.0
    if eps:
        w = w * (1 - eps) + eps / P
    streams = x.view(x.shape[0], hidden, P).permute(0, 2, 1)  # (s, P, hidden)
    streams = streams[:-1]                    # positions that predict a next token
    targets = ids[0, 1:]
    s = streams.shape[0]

    combos = {k: list(itertools.combinations(range(P), k)) for k in range(1, P + 1)}
    subset_sum = {k: [0.0] * len(combos[k]) for k in combos}  # summed token NLL per subset
    oracle_sum = 0.0
    gate_sum = 0.0
    gate_weights = w[:-1]

    chunk = 32
    with torch.no_grad():
        for start in range(0, s, chunk):
            stop = min(start + chunk, s)
            hs = streams[start:stop]          # (c, P, hidden)
            tg = targets[start:stop]
            logits = torch.stack([model.lm_head(hs[:, i, :]) for i in range(P)], dim=1)
            n_tok = stop - start
            for k, cs in combos.items():
                for j, comb in enumerate(cs):
                    subset_sum[k][j] += subset_nll(logits, tg, list(comb)) * n_tok
            # per-token oracle: best stream for each position
            lp = logits.float().log_softmax(-1)
            tok_nll = -lp.gather(2, tg[:, None, None].expand(-1, P, 1)).squeeze(-1)
            oracle_sum += float(tok_nll.min(dim=1).values.sum())
            # learned gate acts on hidden states, so on logits it is linear
            gw = gate_weights[start:stop]
            g_logits = (logits * gw.unsqueeze(-1)).sum(dim=1)
            gate_sum += float(-g_logits.float().log_softmax(-1).gather(1, tg[:, None]).sum())

    subset_nlls = {k: [v / s for v in subset_sum[k]] for k in combos}
    curve = {}
    for k, vals in subset_nlls.items():
        best = min(range(len(vals)), key=vals.__getitem__)
        worst = max(range(len(vals)), key=vals.__getitem__)
        curve[str(k)] = {"mean": sum(vals) / len(vals), "min": vals[best], "max": vals[worst],
                         "argmin": list(combos[k][best]), "argmax": list(combos[k][worst]),
                         "n_subsets": len(vals)}
    out = {
        "model": Path(args.model).name,
        "device": args.device,
        "dtype": "bfloat16",
        "tokens": int(s + 1),
        "predicted_positions": int(s),
        "per_stream_nll": subset_nlls[1],
        "subset_curve": curve,
        "subset_nll": {str(k): [{"streams": list(c), "nll": v} for c, v in zip(combos[k], subset_nlls[k])]
                       for k in combos},
        "oracle_nll": oracle_sum / s,
        "gate_nll": gate_sum / s,
        "mean_gate_weight": w.mean(dim=0).tolist(),
        "provenance": provenance(args.model, args.device),
        "script_sha256": sha256(__file__),
        "text_sha256": hashlib.sha256(TEXT.encode('utf-8')).hexdigest(),
    }
    text = json.dumps(out, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
