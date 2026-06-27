#!/usr/bin/env python3
"""Diff per-layer hidden_states dumps between eager and graph mode."""
import argparse
import os
import sys

import torch


def load_pt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dump_dir")
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--eager-prefix", default="eager")
    p.add_argument("--graph-prefix", default="graph")
    p.add_argument("--num-slots", type=int, default=None)
    p.add_argument("--threshold", type=float, default=1e-3)
    args = p.parse_args()

    if args.num_slots is None:
        files = [f for f in os.listdir(args.dump_dir)
                 if f.startswith(f"{args.eager_prefix}_step{args.step}_slot")
                 and f.endswith(".pt")]
        if not files:
            print(f"ERROR: no eager files for step {args.step}")
            return 2
        slot_nums = []
        for f in files:
            try:
                s = int(f.split("_slot")[1].split(".")[0])
                slot_nums.append(s)
            except (IndexError, ValueError):
                pass
        args.num_slots = max(slot_nums) + 1
        print(f"Auto-detected num_slots={args.num_slots}\n")

    print(f"Diffing step={args.step}, {args.num_slots} slots, "
          f"threshold={args.threshold:.1e}\n")
    print(f"{'slot':>5}  {'role':>16}  {'max_abs':>11}  {'mean_abs':>11}  "
          f"{'eager_max':>11}  {'graph_max':>11}  verdict")
    print("-" * 100)

    first_diverge = None
    for slot in range(args.num_slots):
        eager_path = os.path.join(args.dump_dir,
            f"{args.eager_prefix}_step{args.step}_slot{slot}.pt")
        graph_path = os.path.join(args.dump_dir,
            f"{args.graph_prefix}_step{args.step}_slot{slot}.pt")
        if not (os.path.exists(eager_path) and os.path.exists(graph_path)):
            print(f"  {slot:>3}  MISSING")
            continue
        e = load_pt(eager_path).float()
        g = load_pt(graph_path).float()
        if e.shape != g.shape:
            print(f"  {slot:>3}  SHAPE MISMATCH e={list(e.shape)} g={list(g.shape)}")
            continue
        diff = (e - g).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        e_max = e.abs().max().item()
        g_max = g.abs().max().item()
        if slot == 0:
            role = "post-embed"
        elif slot == args.num_slots - 1:
            role = "post-norm"
        else:
            role = f"after-L{slot - 1}"
        verdict = "DIVERGENT" if max_abs > args.threshold else "ok"
        if max_abs > args.threshold and first_diverge is None:
            first_diverge = slot
        print(f"  {slot:>3}  {role:>16}  {max_abs:11.4e}  {mean_abs:11.4e}  "
              f"{e_max:11.4e}  {g_max:11.4e}  {verdict}")
    print("-" * 100)
    if first_diverge is None:
        print("No layer exceeds threshold.")
    else:
        role = ("post-embed" if first_diverge == 0 else
                "post-norm" if first_diverge == args.num_slots - 1 else
                f"after layer {first_diverge - 1}")
        print(f"FIRST DIVERGENT SLOT: {first_diverge}  ({role})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
