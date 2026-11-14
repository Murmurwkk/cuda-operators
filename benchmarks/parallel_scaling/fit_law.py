"""Refit ParScale's practical and theoretical scaling laws from Tables 2/8/9.

Data: 24 training runs per dataset (P in {1,2,4,8}, six model sizes), 42B tokens.
The practical law is L = (A / (N * (1 + k ln P)))^alpha + E.
The theoretical law is L = (A / (N * [P / ((P-1) rho + 1)]**(1/alpha)))**alpha + E.
"""

import json
import argparse
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

OUT = Path(__file__).with_name("results")

# Non-embedding parameter counts, 0.5B through 4.4B (paper, Table 2 / appendix).
N = np.array([535_813_376, 693_753_856, 1_088_376_320, 1_571_472_384, 2_774_773_760, 4_353_203_200], dtype=float)
P = np.array([1, 2, 4, 8], dtype=float)

STACK = np.array(
    [
        [1.1722, 1.1496, 1.1131, 1.0817, 1.0451, 1.0213],
        [1.1507, 1.1262, 1.0940, 1.0623, 1.0244, 1.0025],
        [1.1354, 1.1124, 1.0808, 1.0490, 1.0126, 0.9906],
        [1.1231, 1.0997, 1.0688, 1.0383, 1.0016, 0.9794],
    ]
)
PILE = np.array(
    [
        [2.1113, 2.0671, 2.0027, 1.9539, 1.8876, 1.8451],
        [2.0772, 2.0363, 1.9730, 1.9266, 1.8610, 1.8137],
        [2.0544, 2.0128, 1.9509, 1.9040, 1.8394, 1.7938],
        [2.0364, 1.9933, 1.9318, 1.8856, 1.8218, 1.7772],
    ]
)

# Published fit, for comparison.
PAPER = {
    "stack": dict(A=1.130616e7, k=0.393463, E=0.691237, alpha=0.189371),
    "pile": dict(A=1.973520e8, k=0.334456, E=1.288766, alpha=0.196333),
    "stack_theo": dict(A=1.187646e7, rho=0.891914, E=0.660016, alpha=0.175036),
    "pile_theo": dict(A=2.150890e8, rho=0.899475, E=1.272178, alpha=0.190100),
}


def practical(theta, n, p):
    a, k, e, alpha = theta
    return (a / (n * (1.0 + k * np.log(p)))) ** alpha + e


def theoretical(theta, n, p):
    a, rho, e, alpha = theta
    diversity = (p / ((p - 1.0) * rho + 1.0)) ** (1.0 / alpha)
    return (a / (n * diversity)) ** alpha + e


def residual(theta, n, p, y, fn):
    return np.log(fn(theta, n, p)) - np.log(y)


def fit(n, p, y, fn, starts, bounds):
    best = None
    for x0 in starts:
        try:
            r = least_squares(residual, x0, args=(n, p, y, fn), bounds=bounds, loss="huber", f_scale=1e-3,
                              max_nfev=4000)
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


def grid_starts():
    out = []
    for e in (math.exp(-1), math.exp(-0.5), 1.0):
        for ai in range(-4, 5):
            for alpha in (0.05, 0.5, 1.0, 1.5, 2.0):
                for k in (0.2, 0.4, 0.6, 0.8):
                    out.append([math.exp(ai) * 1e9, k, e, alpha])
    return out


def r2(pred, true):
    return 1.0 - np.sum((pred - true) ** 2) / np.sum((true - true.mean()) ** 2)


def bootstrap(n, p, y, theta, fn, bounds, reps=500, seed=0):
    rng = np.random.default_rng(seed)
    pred = fn(theta, n, p)
    res = np.log(pred) - np.log(y)
    out = []
    for _ in range(reps):
        ystar = pred * np.exp(rng.choice(res, size=res.size, replace=True))
        r = fit(n, p, ystar, fn, [theta], bounds)
        if r is not None:
            out.append(r.x)
    return np.array(out)


def summarize(name, theta, n, p, y):
    pred = practical(theta, n, p)
    a, k, e, alpha = theta
    return {
        "fit": "practical",
        "A": a, "k": k, "E": e, "alpha": alpha,
        "r2": r2(pred, y),
        "max_abs_err": float(np.max(np.abs(pred - y))),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "multiplier": {str(int(pi)): float(1 + k * math.log(pi)) for pi in (2, 4, 8, 16, 32, 64)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT / 'fit.json')
    args = parser.parse_args()
    results = {}
    for name, table in (("stack", STACK), ("pile", PILE)):
        n = np.repeat(N, 4)
        p = np.tile(P, 6)
        y = table.T.reshape(-1)
        r = fit(n, p, y, practical, grid_starts() + [
                    [PAPER[name]["A"], PAPER[name]["k"], PAPER[name]["E"], PAPER[name]["alpha"]]],
                ([1e4, 0.01, 0.0, 0.02], [1e11, 2.0, 3.0, 1.0]))
        print(name, "practical", r.x, "huber", r.cost / y.size, "r2", r2(practical(r.x, n, p), y))
        bs = bootstrap(n, p, y, r.x, practical,
                       ([1e4, 0.01, 0.0, 0.02], [1e11, 2.0, 3.0, 1.0]), reps=400)
        entry = summarize(name, r.x, n, p, y)
        entry["boot_ci"] = {
            "k": np.percentile(bs[:, 1], [2.5, 97.5]).tolist(),
            "alpha": np.percentile(bs[:, 3], [2.5, 97.5]).tolist(),
            "E": np.percentile(bs[:, 2], [2.5, 97.5]).tolist(),
        }
        entry["boot_mult8"] = np.percentile(1 + bs[:, 1] * math.log(8), [2.5, 97.5]).tolist()
        # Parameter-scaling plus stream overhead, from the appendix table.
        n16 = 1_595_669_000.0  # 1.6B, P=8 including prefixes/aggregator
        n44 = 4_353_203_200.0
        target = n44 / 1_571_472_384.0
        entry["p_star_for_4.4B"] = float(math.exp((target - 1) / r.x[1]))
        entry["loss_1.6B_P8"] = float(practical(r.x, n16, 8.0))
        entry["loss_4.4B_P1"] = float(practical(r.x, n44, 1.0))
        entry["gain_by_N"] = [
            {"N": float(ni), "P1": float(yi[0]), "P8": float(yi[3]),
             "abs": float(yi[0] - yi[3]), "rel": float((yi[0] - yi[3]) / yi[0])}
            for ni, yi in zip(N, table.T)
        ]
        results[name] = entry

        # Theoretical form.
        starts = []
        for ai in range(-2, 5):
            for rho in (0.3, 0.6, 0.9, 0.99):
                for e in (math.exp(-1), 1.0):
                    starts.append([math.exp(ai) * 1e8, rho, e, 0.2])
        pt = PAPER[name + "_theo"]
        starts.append([pt["A"], pt["rho"], pt["E"], pt["alpha"]])
        rt = fit(n, p, y, theoretical, starts, ([1e4, 0.0, 0.0, 0.02], [1e11, 0.999, 3.0, 1.0]))
        print(name, "theoretical", rt.x, "huber", rt.cost / y.size,
              "r2", r2(theoretical(rt.x, n, p), y))
        results[name]["theoretical"] = {
            "A": rt.x[0], "rho": rt.x[1], "E": rt.x[2], "alpha": rt.x[3],
            "r2": r2(theoretical(rt.x, n, p), y),
            "max_abs_err": float(np.max(np.abs(theoretical(rt.x, n, p) - y))),
            "mult8": float((8.0 / (7.0 * rt.x[1] + 1.0)) ** (1.0 / rt.x[3])),
        }

    # Predicted table for figure reproduction.
    n = np.repeat(N, 4)
    p = np.tile(P, 6)
    for name, table in (("stack", STACK), ("pile", PILE)):
        th = results[name]
        pred = practical([th["A"], th["k"], th["E"], th["alpha"]], n, p)
        results[name]["pred_table"] = pred.reshape(6, 4).T.tolist()

    results["paper"] = PAPER
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("pred_table", "gain_by_N")}
                      for k, v in results.items() if k != "paper"}, indent=2))


if __name__ == "__main__":
    main()
