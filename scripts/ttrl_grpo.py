#!/usr/bin/env python3
"""Standalone TTRL (arXiv:2504.16084) GRPO trainer — single GPU, HF transformers.

Faithful to the paper's core mechanism (which the authors describe as "a simple
reward-function modification"):

  1. For each prompt, sample G rollouts from the current policy.
  2. Extract answers; take the MAJORITY-VOTE answer as the pseudo-label
     (NO ground-truth labels are used to train).
  3. Binary reward: 1 if a rollout's answer == pseudo-label, else 0.
  4. GRPO update: advantage = (r - mean_group(r)) / std_group(r); maximize
     sum_t logπ(a_t) * A  (token-mean), with a KL-to-ref penalty.

Ground truth is used ONLY to *report* accuracy (label_accuracy / pass@1), never
to compute the training reward — exactly as in TTRL.

This bypasses verl (whose vLLM rollout is incompatible with vLLM 0.23 for the
qwen3_5 hybrid-mamba arch) while reproducing the same claim.
"""
import argparse, json, os, sys, collections, random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from math_verify import parse as mv_parse, verify as mv_verify


# ---- answer extraction + grading (same convention as the TTRL repo) ----
def extract_answer(text):
    if text is None or "\\boxed" not in text:
        return None
    idx = text.rfind("\\boxed")
    i = text.find("{", idx)
    if i == -1:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
        j += 1
    return None


def equal(a, b):
    if a is None or b is None:
        return False
    a, b = str(a).strip(), str(b).strip()
    if a == b:
        return True
    try:
        return bool(mv_verify(mv_parse("$" + b + "$"), mv_parse("$" + a + "$")))
    except Exception:
        return False


def majority_vote(answers):
    cnt = collections.Counter([a for a in answers if a is not None])
    if not cnt:
        return None, 0.0
    ans, c = cnt.most_common(1)[0]
    return ans, c / len(answers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)        # train json
    ap.add_argument("--out", required=True)          # HF save dir
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--group-size", type=int, default=8)   # rollouts per prompt (G)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--kl-coef", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    dev = "cuda"
    os.makedirs(args.artifacts, exist_ok=True)

    # cuDNN SDPA backend rejects qwen3_5's head_dim=256 ("No valid execution
    # plans built"); force eager attention + disable cuDNN SDPA to be safe.
    torch.backends.cuda.enable_cudnn_sdp(False)
    attn_impl = "eager"

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16,
        attn_implementation=attn_impl).to(dev)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    # frozen reference for KL
    ref = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16,
        attn_implementation=attn_impl).to(dev)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    data = json.load(open(args.data))
    def fmt(d):
        msg = [{"role": "user", "content": d["prompt"] +
                "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
        return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    log = []
    G = args.group_size
    for step in range(args.steps):
        batch = random.sample(data, min(args.prompts_per_step, len(data)))
        prompts = [fmt(d) for d in batch]
        gts = [str(d["answer"]) for d in batch]

        # ---- 1. sample G rollouts per prompt ----
        model.eval()
        all_seq, all_promptlen, group_rewards, group_meta = [], [], [], []
        with torch.no_grad():
            for pi, ptext in enumerate(prompts):
                enc = tok(ptext, return_tensors="pt").to(dev)
                plen = enc.input_ids.shape[1]
                gen = model.generate(
                    **enc, do_sample=True, temperature=args.temperature, top_p=0.95,
                    max_new_tokens=args.max_new_tokens, num_return_sequences=G,
                    pad_token_id=tok.pad_token_id)
                texts = tok.batch_decode(gen[:, plen:], skip_special_tokens=True)
                answers = [extract_answer(t) for t in texts]
                # ---- 2. majority-vote pseudo-label (NO ground truth) ----
                pseudo, ratio = majority_vote(answers)
                # ---- 3. binary reward vs pseudo-label ----
                rewards = [1.0 if (a is not None and equal(a, pseudo)) else 0.0 for a in answers]
                # reporting only (uses GT, not for training):
                gt_acc = sum(1.0 for a in answers if equal(a, gts[pi])) / G
                label_ok = 1.0 if equal(pseudo, gts[pi]) else 0.0
                for gi in range(G):
                    all_seq.append(gen[gi])
                    all_promptlen.append(plen)
                group_rewards.append(rewards)
                group_meta.append({"pseudo": pseudo, "ratio": ratio,
                                    "gt_acc": gt_acc, "label_ok": label_ok,
                                    "mean_r": sum(rewards) / G})

        # ---- 4. GRPO advantages (group-normalized) ----
        advs = []
        for rewards in group_rewards:
            r = torch.tensor(rewards)
            a = (r - r.mean()) / (r.std() + 1e-6)
            advs.extend(a.tolist())

        # ---- policy gradient update (token-mean logp * adv + KL penalty) ----
        model.train()
        opt.zero_grad()
        total_loss, total_kl, ntok = 0.0, 0.0, 0
        for idx in range(len(all_seq)):
            seq = all_seq[idx].unsqueeze(0).to(dev)
            plen = all_promptlen[idx]
            adv = advs[idx]
            if abs(adv) < 1e-8:
                continue
            attn = (seq != tok.pad_token_id).long()
            out = model(seq, attention_mask=attn)
            logits = out.logits[:, :-1, :]
            tgt = seq[:, 1:]
            logp = torch.log_softmax(logits.float(), dim=-1).gather(
                -1, tgt.unsqueeze(-1)).squeeze(-1)
            resp_mask = torch.zeros_like(tgt, dtype=torch.float)
            resp_mask[:, plen - 1:] = 1.0
            resp_mask = resp_mask * attn[:, 1:].float()
            denom = resp_mask.sum().clamp(min=1.0)
            pg = -(adv * (logp * resp_mask).sum() / denom)
            kl = torch.tensor(0.0, device=dev)
            if args.kl_coef > 0:
                with torch.no_grad():
                    rlogits = ref(seq, attention_mask=attn).logits[:, :-1, :]
                    rlogp = torch.log_softmax(rlogits.float(), dim=-1).gather(
                        -1, tgt.unsqueeze(-1)).squeeze(-1)
                kl = ((logp - rlogp) * resp_mask).sum() / denom
            loss = (pg + args.kl_coef * kl) / len(all_seq)
            loss.backward()
            total_loss += pg.item(); total_kl += kl.item(); ntok += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        mr = sum(m["mean_r"] for m in group_meta) / len(group_meta)
        lacc = sum(m["label_ok"] for m in group_meta) / len(group_meta)
        gacc = sum(m["gt_acc"] for m in group_meta) / len(group_meta)
        rat = sum(m["ratio"] for m in group_meta) / len(group_meta)
        rec = {"step": step, "mean_reward": round(mr, 3),
               "label_accuracy": round(lacc, 3), "gt_pass@1": round(gacc, 3),
               "majority_ratio": round(rat, 3), "pg_loss": round(total_loss / max(1, ntok), 4)}
        log.append(rec)
        print(f"[ttrl step {step}] reward={rec['mean_reward']} "
              f"label_acc={rec['label_accuracy']} gt_pass@1={rec['gt_pass@1']} "
              f"maj_ratio={rec['majority_ratio']}", flush=True)
        json.dump(log, open(os.path.join(args.artifacts, "train_log.json"), "w"), indent=2)

    model.config.use_cache = True
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("saved TTRL model to", args.out)


if __name__ == "__main__":
    main()
