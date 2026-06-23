#!/usr/bin/env python3
"""Smoke gate: can the upgraded stack load + serve qwen3_5?

Must be a real file (NOT a heredoc/`python -`): vLLM's V1 engine spawns worker
subprocesses that re-import __main__, which fails for <stdin>. Guarded by
`if __name__ == "__main__"` so spawn re-import is safe.
"""
import sys, traceback


def main():
    model_id = sys.argv[1]
    ok = True
    # (a) transformers knows the arch?
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print("transformers config OK:", type(cfg).__name__,
              "model_type=", getattr(cfg, "model_type", "?"))
    except Exception:
        ok = False; traceback.print_exc()
    # (b) vLLM importable / version?
    try:
        import vllm
        print("vllm version:", vllm.__version__)
        from vllm import LLM, SamplingParams
    except Exception:
        ok = False; traceback.print_exc()
    # (c) actually load + generate (the real gate)
    if ok:
        try:
            llm = LLM(model=model_id, trust_remote_code=True,
                      max_model_len=2048, gpu_memory_utilization=0.85,
                      enforce_eager=True, dtype="bfloat16")
            out = llm.generate(["What is 12*12? Answer:"],
                               SamplingParams(temperature=0.0, max_tokens=32))
            print("GEN_OK:", repr(out[0].outputs[0].text[:120]))
            print("SMOKE_PASS")
        except Exception:
            ok = False; traceback.print_exc()
    sys.exit(0 if ok else 17)


if __name__ == "__main__":
    main()
