#!/usr/bin/env python3
"""Standalone vLLM eval for TTRL repro: pass@1 (avg@n) and maj@n on a math set.

Reuses the repo's math grader + majority-vote answer extraction so the numbers
are computed exactly the same way TTRL computes its pseudo-labels / rewards.
"""
import argparse, json, os, re, collections

# Self-contained grader using math_verify (same library the TTRL repo uses for
# grading) so eval does NOT import the heavy `verl` package (pandas/tensordict).
from math_verify import parse as mv_parse, verify as mv_verify


def extract_answer(passage):
    if passage is None or "\\boxed" not in passage:
        return None
    # extract the content of the LAST \boxed{...}, balancing braces
    idx = passage.rfind("\\boxed")
    i = passage.find("{", idx)
    if i == -1:
        return None
    depth, j = 0, i
    while j < len(passage):
        if passage[j] == "{":
            depth += 1
        elif passage[j] == "}":
            depth -= 1
            if depth == 0:
                return passage[i + 1:j]
        j += 1
    return None


def grade(model_answer, gt_answer):
    if model_answer is None:
        return False
    ma, gt = str(model_answer).strip(), str(gt_answer).strip()
    if ma == gt:
        return True
    try:
        # math_verify compares mathematically (handles 025==25, fractions, etc.)
        return bool(mv_verify(mv_parse("$" + gt + "$"), mv_parse("$" + ma + "$")))
    except Exception:
        return False


def majority(answers):
    answers = [a for a in answers if a is not None]
    if not answers:
        return None, 0.0
    c = collections.Counter(answers)
    ans, cnt = c.most_common(1)[0]
    return ans, cnt / max(1, len(answers))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    data = json.load(open(args.data))
    prompts = [
        d["prompt"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        for d in data
    ]
    gts = [str(d["answer"]) for d in data]

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, trust_remote_code=True, max_model_len=args.max_tokens + 1024,
              gpu_memory_utilization=0.85, enforce_eager=False, dtype="bfloat16")
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens)
    outs = llm.generate(prompts, sp)

    pass1_sum = 0.0   # avg@n: mean per-sample correctness
    maj_correct = 0   # maj@n: majority-voted answer correct
    per = []
    for i, o in enumerate(outs):
        texts = [c.text for c in o.outputs]
        answers = [extract_answer(t) for t in texts]
        # per-sample accuracy (avg@n == pass@1 in TTRL tables)
        accs = [1.0 if (a is not None and grade(a, gts[i])) else 0.0 for a in answers]
        avg = sum(accs) / len(accs)
        pass1_sum += avg
        maj_ans, maj_ratio = majority(answers)
        maj_ok = 1 if (maj_ans is not None and grade(maj_ans, gts[i])) else 0
        maj_correct += maj_ok
        per.append({"idx": i, "gt": gts[i], "avg@n": avg, "maj_ans": maj_ans,
                    "maj_ratio": maj_ratio, "maj_ok": maj_ok})

    N = len(prompts)
    res = {
        "label": args.label, "model": args.model, "n_samples": args.n, "n_problems": N,
        "pass@1": round(100.0 * pass1_sum / N, 2),
        f"maj@{args.n}": round(100.0 * maj_correct / N, 2),
        "per_problem": per,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[{args.label}] pass@1={res['pass@1']}  maj@{args.n}={res[f'maj@{args.n}']}  (N={N})")


if __name__ == "__main__":
    main()
