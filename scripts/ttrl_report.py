#!/usr/bin/env python3
"""Assemble EVAL.md from baseline/ttrl eval JSONs in the artifacts dir."""
import json, os, sys

art = sys.argv[1] if len(sys.argv) > 1 else ".openresearch/artifacts"

def load(name):
    p = os.path.join(art, name)
    return json.load(open(p)) if os.path.exists(p) else None

base = load("eval_base.json")
ttrl = load("eval_ttrl.json")
tlog = load("train_log.json")

lines = ["# TTRL reproduction — Qwen/Qwen3.5-0.8B-Base on AIME\n"]
lines.append("Test-Time RL (arXiv:2504.16084): majority-vote pseudo-labels -> GRPO, "
             "**no ground-truth labels** used in training. Fresh base (non-instruct) model.\n")

if base is None:
    lines.append("\n**Status:** no baseline eval produced yet — see status.txt / smoke.txt.\n")
else:
    nkey = [k for k in base if k.startswith("maj@")][0]
    lines.append("\n## Results (AIME-TTT, pass@1 = avg@n)\n")
    lines.append("| Method | pass@1 | " + nkey + " |")
    lines.append("|---|---|---|")
    lines.append(f"| Base (no TTRL) | {base['pass@1']} | {base[nkey]} |")
    if ttrl is not None:
        nk2 = [k for k in ttrl if k.startswith("maj@")][0]
        lines.append(f"| **TTRL** | **{ttrl['pass@1']}** | {ttrl[nk2]} |")
        d = ttrl['pass@1'] - base['pass@1']
        rel = (d / base['pass@1'] * 100) if base['pass@1'] > 0 else float('inf')
        lines.append("")
        lines.append(f"**Margin (pass@1): {base['pass@1']} -> {ttrl['pass@1']} "
                     f"= +{round(d,2)} pts ({'+%.0f%%' % rel if rel!=float('inf') else 'inf'} relative).**")
        verdict = "HUGE MARGIN — reproduced" if d >= 5 else ("positive" if d > 0 else "no improvement")
        lines.append(f"\n**Verdict:** {verdict}.")
    else:
        lines.append("\n_TTRL-trained eval not present yet (training stage incomplete)._")

if tlog:
    lines.append("\n## TTRL training curve (label-free signal)\n")
    lines.append("| step | mean_reward | label_accuracy | gt_pass@1 | maj_ratio |")
    lines.append("|---|---|---|---|---|")
    show = tlog[:: max(1, len(tlog) // 12)]
    for r in show:
        lines.append(f"| {r['step']} | {r.get('mean_reward')} | "
                     f"{r.get('label_accuracy')} | {r.get('gt_pass@1')} | {r.get('majority_ratio')} |")
    first, last = tlog[0], tlog[-1]
    dg = last.get("gt_pass@1", 0) - first.get("gt_pass@1", 0)
    lines.append(f"\nTrain gt_pass@1: {first.get('gt_pass@1')} -> {last.get('gt_pass@1')} "
                 f"(Δ {round(dg,3)}) over {len(tlog)} steps.")

print("\n".join(lines))
with open(os.path.join(art, "EVAL.md"), "w") as f:
    f.write("\n".join(lines) + "\n")
