"""
sd3/run_eval.py  —  Multi-GPU evaluation runner for TCSA on Stable Diffusion 3
================================================================================
Analogous to scripts/evaluation/merged_unlearning_multi_gpu.py but wired to the
SD3 TCSA pipeline (JointTransformerBlock hooks + T5 orthogonalisation).

Benchmarks (Table 4):
  p4d   — datasets/nudity.csv             (col: prompt, per-row seed/guidance)
  ring  — datasets/nudity-ring-a-bell.csv (col: sensitive prompt)
  mma   — datasets/mma-diffusion-nsfw-adv-prompts.csv (col: adv_prompt)

All three produce {i}_0.png output compatible with q16_nudenet_detect.py / eval_asr.py.

─── Workflow ──────────────────────────────────────────────────────────────────

# Step 1 — build steering vectors once (cached):
python sd3/run_eval.py build-vectors \\
    --save_path steering_vectors/sv_sd3_nudity.pt \\
    --device 0

# Step 2 — generate for each benchmark:
python sd3/run_eval.py eval --benchmark p4d \\
    --sv_path steering_vectors/sv_sd3_nudity.pt \\
    --save_dir results/sd3_tcsa/p4d  --num_gpus 1

python sd3/run_eval.py eval --benchmark ring \\
    --sv_path steering_vectors/sv_sd3_nudity.pt \\
    --save_dir results/sd3_tcsa/ring --num_gpus 1

python sd3/run_eval.py eval --benchmark mma \\
    --sv_path steering_vectors/sv_sd3_nudity.pt \\
    --save_dir results/sd3_tcsa/mma  --num_gpus 1

# Step 3 — evaluate ASR (existing evaluators, unchanged):
python scripts/evaluation/q16_nudenet_detect.py \\
    --folder_path results/sd3_tcsa/p4d \\
    --prompts_path datasets/nudity.csv
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
from typing import List, Optional, Tuple

import torch
import pandas as pd

# ── find tcsa_sd3.py (lives in scripts/evaluation/) ──────────────────────────
_SD3_DIR   = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SD3_DIR)
_EVAL_DIR  = os.path.join(_REPO_ROOT, "scripts", "evaluation")
for _p in (_EVAL_DIR, _SD3_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tcsa_sd3 import (          # noqa: E402
    TCSA_SD3,
    Schedule,
    build_steering_vectors_sd3,
    make_comparison,
    make_grid_row,
)

log = logging.getLogger("run_eval_sd3")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SD3_MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"

_DATASETS = {
    "p4d":  os.path.join(_REPO_ROOT, "datasets", "nudity.csv"),
    "ring": os.path.join(_REPO_ROOT, "datasets", "nudity-ring-a-bell.csv"),
    "mma":  os.path.join(_REPO_ROOT, "datasets", "mma-diffusion-nsfw-adv-prompts.csv"),
}
_PROMPT_COL = {
    "p4d":  "prompt",
    "ring": "sensitive prompt",
    "mma":  "adv_prompt",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_dataset(
    csv_path: str,
    prompt_col: str,
    seed_default: int = 42,
    guidance_default: float = 7.0,
    n: Optional[int] = None,
) -> Tuple[List[int], List[str], List[int], List[float]]:
    """Return (indices, prompts, seeds, guidances).

    Uses per-row evaluation_seed / evaluation_guidance when available
    (p4d format); falls back to defaults for ring / mma.
    """
    df = pd.read_csv(csv_path)

    if prompt_col not in df.columns:
        raise ValueError(
            f"Column '{prompt_col}' not found in {csv_path}.\n"
            f"Available: {df.columns.tolist()}"
        )

    df = df[df[prompt_col].notna() & (df[prompt_col].astype(str).str.strip() != "")]
    if n is not None:
        df = df.iloc[:n]

    indices   = list(df.index)
    prompts   = list(df[prompt_col].astype(str))
    seeds     = (list(df["evaluation_seed"].astype(int))
                 if "evaluation_seed" in df.columns
                 else [seed_default] * len(prompts))
    guidances = (list(df["evaluation_guidance"].astype(float))
                 if "evaluation_guidance" in df.columns
                 else [guidance_default] * len(prompts))

    return indices, prompts, seeds, guidances


def _load_sv(
    sv_path: str,
    model_id: str,
    device: str,
    num_pairs: int,
    num_steps: int,
) -> tuple:
    """Load cached or build new SD3 steering vectors.
    Returns (sv, layer_align, t5_direction).
    """
    if os.path.exists(sv_path):
        data = torch.load(sv_path, map_location="cpu", weights_only=False)
        t5_dir = data.get("t5_direction")
        if t5_dir is None:
            raise RuntimeError(
                "Cache file missing 't5_direction'. "
                "Delete the .pt file and re-run build-vectors."
            )
        log.info("Loaded steering vectors from %s", sv_path)
        return data["sv"], data["layer_align"], t5_dir

    return build_steering_vectors_sd3(
        model_id=model_id,
        num_pairs=num_pairs,
        save_path=sv_path,
        device=device,
        num_steps=num_steps,
        seed=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Single-GPU worker
# ─────────────────────────────────────────────────────────────────────────────

def _worker(
    rank: int,
    indices: List[int],
    prompts: List[str],
    seeds: List[int],
    guidances: List[float],
    sv_path: str,
    save_dir: str,
    model_id: str,
    num_pairs: int,
    num_steps: int,
    image_size: int,
    beta: float,
    max_proj_ratio: float,
    dtype_str: str,
) -> None:
    """
    Generates one TCSA (mode=both) image per prompt on cuda:{rank}.
    Saves results as {index}_0.png — the format expected by q16_nudenet_detect.py.
    """
    from tqdm import tqdm

    device = f"cuda:{rank}"
    dtype  = {"bfloat16": torch.bfloat16,
               "float16":  torch.float16,
               "float32":  torch.float32}[dtype_str]

    log.info("[GPU %d] %d prompts  device=%s", rank, len(indices), device)

    sv, layer_align, t5_dir = _load_sv(sv_path, model_id, device, num_pairs, num_steps)

    wrapper = TCSA_SD3.from_pretrained(
        model_id=model_id,
        sv=sv,
        layer_align=layer_align,
        t5_direction=t5_dir,
        device=device,
        beta=beta,
        max_proj_ratio=max_proj_ratio,
        torch_dtype=dtype,
    )
    schedule = Schedule()  # default front-loaded schedule

    for global_i, prompt, seed, guidance in tqdm(
        zip(indices, prompts, seeds, guidances),
        total=len(indices),
        desc=f"GPU {rank}",
        position=rank,
        leave=True,
    ):
        out_path = os.path.join(save_dir, f"{global_i}_0.png")
        if os.path.exists(out_path):
            continue

        img = wrapper.generate(
            prompt=prompt,
            mode="both",
            schedule=schedule,
            seed=int(seed),
            guidance_scale=float(guidance),
            num_steps=num_steps,
            height=image_size,
            width=image_size,
        )
        img.save(out_path)

    wrapper.remove_hooks()
    del wrapper, sv, layer_align, t5_dir
    gc.collect()
    torch.cuda.empty_cache()
    log.info("[GPU %d] done.", rank)


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-GPU dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(
    csv_path: str,
    prompt_col: str,
    sv_path: str,
    save_dir: str,
    model_id: str = SD3_MODEL_ID,
    num_gpus: int = 1,
    num_pairs: int = 50,
    num_steps: int = 28,
    image_size: int = 512,
    beta: float = 2.0,
    max_proj_ratio: float = 0.5,
    seed_default: int = 42,
    guidance_default: float = 7.0,
    dtype_str: str = "bfloat16",
    n: Optional[int] = None,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    indices, prompts, seeds, guidances = _load_dataset(
        csv_path, prompt_col, seed_default, guidance_default, n
    )
    total = len(indices)
    log.info("Dataset    : %s  (%d prompts)", os.path.basename(csv_path), total)
    log.info("Save dir   : %s", save_dir)
    log.info("GPUs: %d  steps: %d  size: %d  beta: %.1f", num_gpus, num_steps, image_size, beta)

    worker_kw = dict(
        sv_path=sv_path, save_dir=save_dir, model_id=model_id,
        num_pairs=num_pairs, num_steps=num_steps, image_size=image_size,
        beta=beta, max_proj_ratio=max_proj_ratio, dtype_str=dtype_str,
    )

    if num_gpus == 1:
        _worker(rank=0, indices=indices, prompts=prompts,
                seeds=seeds, guidances=guidances, **worker_kw)
    else:
        import torch.multiprocessing as mp

        idx_chunks = [indices[i::num_gpus]   for i in range(num_gpus)]
        pmt_chunks = [prompts[i::num_gpus]   for i in range(num_gpus)]
        sd_chunks  = [seeds[i::num_gpus]     for i in range(num_gpus)]
        gd_chunks  = [guidances[i::num_gpus] for i in range(num_gpus)]

        for rank, ci in enumerate(idx_chunks):
            log.info("  GPU %d -> %d prompts", rank, len(ci))

        mp.set_start_method("spawn", force=True)
        procs = []
        for rank in range(num_gpus):
            p = mp.Process(
                target=_worker,
                kwargs=dict(rank=rank, indices=idx_chunks[rank],
                            prompts=pmt_chunks[rank], seeds=sd_chunks[rank],
                            guidances=gd_chunks[rank], **worker_kw),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        failed = [p.exitcode for p in procs if p.exitcode != 0]
        if failed:
            raise RuntimeError(f"{len(failed)} worker(s) failed: {failed}")

    done = sum(
        1 for i in indices
        if os.path.exists(os.path.join(save_dir, f"{i}_0.png"))
    )
    print("\n" + "=" * 60)
    print(f"Done.  {done}/{total} images -> {save_dir}")
    print("\nNext — evaluate ASR:")
    print(f"  python scripts/evaluation/q16_nudenet_detect.py \\")
    print(f"      --folder_path {save_dir} \\")
    print(f"      --prompts_path {csv_path}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _add_common(sp) -> None:
    sp.add_argument("--model_id",  default=SD3_MODEL_ID)
    sp.add_argument("--sv_path",   required=True)
    sp.add_argument("--num_pairs", type=int,   default=50)
    sp.add_argument("--steps",     type=int,   default=28)
    sp.add_argument("--img_size",  type=int,   default=512)
    sp.add_argument("--beta",      type=float, default=2.0)
    sp.add_argument("--max_proj_ratio", type=float, default=0.5)
    sp.add_argument("--dtype",     default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    sp.add_argument("--device",    default="0",
                    help="GPU index for single-GPU commands.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TCSA-SD3 — build vectors, generate images, fill Table 4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # build-vectors ──────────────────────────────────────────────────────────
    bv = sub.add_parser("build-vectors",
                        help="Precompute and cache SD3 steering vectors.")
    bv.add_argument("--model_id",  default=SD3_MODEL_ID)
    bv.add_argument("--num_pairs", type=int, default=50)
    bv.add_argument("--save_path", required=True)
    bv.add_argument("--device",    default="0")
    bv.add_argument("--steps",     type=int, default=28)
    bv.add_argument("--seed",      type=int, default=0)

    # eval ───────────────────────────────────────────────────────────────────
    ev = sub.add_parser("eval",
                        help="Batch generate {i}_0.png for ASR scoring.")
    ev.add_argument("--benchmark", choices=["p4d", "ring", "mma"], default=None,
                    help="Preset: p4d | ring | mma. Overrides --prompts/--col.")
    ev.add_argument("--prompts",   default=None,
                    help="CSV path for custom dataset (use with --col).")
    ev.add_argument("--col",       default="prompt")
    ev.add_argument("--save_dir",  required=True)
    ev.add_argument("--num_gpus",  type=int, default=1)
    ev.add_argument("--n",         type=int, default=None,
                    help="Limit to first N prompts.")
    ev.add_argument("--seed",      type=int,   default=42)
    ev.add_argument("--guidance",  type=float, default=7.0)
    _add_common(ev)

    # single ─────────────────────────────────────────────────────────────────
    sg = sub.add_parser("single",
                        help="4-column comparison grid for one prompt.")
    sg.add_argument("--prompt",    required=True)
    sg.add_argument("--save_path", default="sd3_tcsa_comparison.png")
    sg.add_argument("--seed",      type=int,   default=42)
    sg.add_argument("--guidance",  type=float, default=7.0)
    _add_common(sg)

    args = parser.parse_args()

    if args.cmd == "build-vectors":
        build_steering_vectors_sd3(
            model_id=args.model_id,
            num_pairs=args.num_pairs,
            save_path=args.save_path,
            device=f"cuda:{int(args.device)}",
            num_steps=args.steps,
            seed=args.seed,
        )

    elif args.cmd == "eval":
        if args.benchmark is not None:
            csv_path   = _DATASETS[args.benchmark]
            prompt_col = _PROMPT_COL[args.benchmark]
        elif args.prompts is not None:
            csv_path   = args.prompts
            prompt_col = args.col
        else:
            parser.error("Provide --benchmark or --prompts.")

        run_eval(
            csv_path=csv_path,
            prompt_col=prompt_col,
            sv_path=args.sv_path,
            save_dir=args.save_dir,
            model_id=args.model_id,
            num_gpus=args.num_gpus,
            num_pairs=args.num_pairs,
            num_steps=args.steps,
            image_size=args.img_size,
            beta=args.beta,
            max_proj_ratio=args.max_proj_ratio,
            seed_default=args.seed,
            guidance_default=args.guidance,
            dtype_str=args.dtype,
            n=args.n,
        )

    elif args.cmd == "single":
        device = f"cuda:{int(args.device)}"
        dtype  = {"bfloat16": torch.bfloat16,
                   "float16":  torch.float16,
                   "float32":  torch.float32}[args.dtype]

        sv, layer_align, t5_dir = _load_sv(
            args.sv_path, args.model_id, device, args.num_pairs, args.steps
        )
        wrapper = TCSA_SD3.from_pretrained(
            model_id=args.model_id,
            sv=sv, layer_align=layer_align, t5_direction=t5_dir,
            device=device, beta=args.beta,
            max_proj_ratio=args.max_proj_ratio,
            torch_dtype=dtype,
        )
        images = make_comparison(
            wrapper, args.prompt, Schedule(),
            seed=args.seed, guidance_scale=args.guidance,
            num_steps=args.steps, height=args.img_size, width=args.img_size,
        )
        row = make_grid_row(images, prompt_text=args.prompt)
        os.makedirs(
            os.path.dirname(os.path.abspath(args.save_path)) or ".",
            exist_ok=True,
        )
        row.save(args.save_path)
        wrapper.remove_hooks()
        print(f"Saved -> {args.save_path}")
