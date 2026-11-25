"""Train and evaluate the forecasters on synthetic rise-and-drain series.

    python -m jalmarg.temporal.train --series 2000 --epochs 20

Splits are by event (series), never by window. Reports MAE of the median at
10/30/60 min, 80% interval coverage, and the time-to-passable unsafe rate for
a two-wheeler threshold. Saves models/forecaster_{gru,hybrid}.pt and
reports/forecaster_results.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .forecaster import (
    HORIZON, GRUForecaster, HybridForecaster, PhysicsForecaster, build_inputs, make_net, pinball_loss,
)
from .synth import generate_dataset, make_windows

HORIZONS = (10, 30, 60)
TWO_WHEELER_THRESHOLD = 18.0  # d_v - m for a two-wheeler (20 - 2 cm)


def _tensors(windows, physics: PhysicsForecaster):
    seqs, rfs, ys, persist, phys = [], [], [], [], []
    for w in windows:
        s, rf = build_inputs(w.h_hist, w.rain_hist, w.rain_future)
        seqs.append(s)
        rfs.append(rf)
        ys.append(w.h_future.astype(np.float32))
        persist.append(np.full(HORIZON, w.h_hist[-1], dtype=np.float32))
        p = physics.fit(w.h_hist, w.rain_hist)
        phys.append(physics.median_path(float(w.h_hist[-1]), w.rain_future, HORIZON, p)[1:].astype(np.float32))
    return [np.stack(a) for a in (seqs, rfs, ys, persist, phys)]


def train_net(hybrid, data, epochs, seed=0, lr=2e-3, batch=128, log=print):
    import torch

    torch.manual_seed(seed)
    seq, rf, y, persist, phys = (torch.from_numpy(a) for a in data)
    base = phys if hybrid else persist
    net = make_net(hybrid)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(y)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            pred = net(seq[idx], rf[idx], base[idx])
            loss = pinball_loss(pred / 10.0, y[idx] / 10.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        log(f"  [{'hybrid' if hybrid else 'gru'}] epoch {ep + 1}/{epochs} pinball={tot / n:.4f}")
    net.eval()
    return net


def evaluate(forecaster, windows):
    errs = {h: [] for h in HORIZONS}
    inside, total = 0, 0
    unsafe, said_safe = 0, 0
    for w in windows:
        q = forecaster.forecast(w.h_hist, w.rain_hist, w.rain_future, sigma_now=1.0, horizon=HORIZON)
        for h in HORIZONS:
            errs[h].append(abs(q.q50[h] - w.h_future[h - 1]))
        inside += int(np.sum((w.h_future >= q.q10[1:]) & (w.h_future <= q.q90[1:])))
        total += HORIZON
        # time-to-passable: first minute q90 < threshold; unsafe if truth is above it then
        ok = np.where(q.q90[1:] < TWO_WHEELER_THRESHOLD)[0]
        if len(ok):
            said_safe += 1
            if w.h_future[ok[0]] >= TWO_WHEELER_THRESHOLD:
                unsafe += 1
    res = {f"mae_{h}min": float(np.mean(errs[h])) for h in HORIZONS}
    res["coverage_80"] = inside / total
    res["ttp_unsafe_rate"] = unsafe / max(said_safe, 1)
    res["ttp_decisions"] = said_safe
    return res


def main(argv=None):
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--series", type=int, default=2000)
    ap.add_argument("--per-series", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--report", default="reports/forecaster_results.json")
    args = ap.parse_args(argv)

    t0 = time.time()
    series = generate_dataset(args.series, seed=args.seed)
    n_test = max(1, len(series) // 5)
    train_s, test_s = series[:-n_test], series[-n_test:]  # split by event
    train_w = make_windows(train_s, args.per_series, seed=args.seed)
    test_w = make_windows(test_s, args.per_series, seed=args.seed + 1)
    print(f"{len(train_s)} train events / {len(test_s)} test events -> "
          f"{len(train_w)} / {len(test_w)} windows")

    physics = PhysicsForecaster()
    data = _tensors(train_w, physics)
    print(f"physics fits for training set done in {time.time() - t0:.0f}s")

    os.makedirs(args.models_dir, exist_ok=True)
    gru_net = train_net(False, data, args.epochs, args.seed)
    hyb_net = train_net(True, data, args.epochs, args.seed)
    gru_path = os.path.join(args.models_dir, "forecaster_gru.pt")
    hyb_path = os.path.join(args.models_dir, "forecaster_hybrid.pt")
    torch.save(gru_net.state_dict(), gru_path)
    torch.save(hyb_net.state_dict(), hyb_path)

    results = {
        "physics": evaluate(physics, test_w),
        "gru": evaluate(GRUForecaster(gru_path), test_w),
        "hybrid": evaluate(HybridForecaster(hyb_path), test_w),
        "config": vars(args) | {"train_windows": len(train_w), "test_windows": len(test_w),
                                "note": "synthetic data only; split by event"},
    }
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'model':8} " + " ".join(f"{k:>14}" for k in results["physics"]))
    for name in ("physics", "gru", "hybrid"):
        print(f"{name:8} " + " ".join(f"{v:>14.3f}" for v in results[name].values()))
    print(f"\nsaved {gru_path}, {hyb_path}, {args.report} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
