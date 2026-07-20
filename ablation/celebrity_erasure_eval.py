#!/usr/bin/env python3
"""
ablation/celebrity_erasure_eval.py — NLCE-style evaluation for TCSA's
celebrity-erasure generations (Table 2 / Figure 4 metric family).

Runs in the ISOLATED `nlce_face` conda env (facenet-pytorch needs an older
torch than the rest of this repo's `casteer` env -- see celebrity_erasure_tcsa.py
docstring for why this is a separate environment).

Reads the baseline/ (mode="none") and steered/ (mode="both", TCSA)
directories produced by `celebrity_erasure_tcsa.py generate` (NLCE's own
joint-portrait fidelity CSVs: "A portrait of {target} and {other}") and
computes:

    Acc_t (down) — % of images where the TARGET celebrity's face is still
                   detected (lower = better erasure)
    Acc_r (up)   — % of images where the OTHER celebrity's face is still
                   correctly detected (higher = better fidelity)
    H_o   (up)   — harmonic mean of erasure-success (1-Acc_t) and
                   retention (Acc_r) -- identical formula to
                   object_erasure_eval.py / NLCE's `Hcc_dec`
    LPIPS_u (down) — LPIPS between baseline and steered, restricted to the
                   region OUTSIDE the target's face bounding box (Figure 4's
                   "LPIPS on non-target regions") -- measures how much the
                   erasure intervention changed everything EXCEPT the target
    CS    (up)   — CLIP score (image vs. the joint-portrait prompt)
    KID   (down) — Kernel Inception Distance, baseline vs steered dirs

Identity classification method (GIPHY's classifier is proprietary and not
available to us): face detection via MTCNN, identity via cosine similarity
of InceptionResnetV1 (VGGFace2-pretrained) embeddings against reference
photos sourced from Wikipedia/Wikimedia Commons (freely licensed) for the
target and the ~50 unique "other" celebrities per campaign.

Face-region masking for LPIPS (NLCE used SAM for pixel-accurate
segmentation; not installed here, substituted with MTCNN's face bounding
box + margin, per user-approved substitution):
    mask = 1.0 everywhere EXCEPT inside the target's face bbox (0.0 there)
    LPIPS computed on (baseline * mask) vs (steered * mask)

New file. Run inside the `nlce_face` conda env:
    /storage/pk_mridul/anaconda3/envs/nlce_face/bin/python ablation/celebrity_erasure_eval.py \\
        --target "Elon Musk" --save_dir ablation/results/celebrity_erasure/elon_musk
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

REF_DIR = "/storage/pk_mridul/TCSA/ablation/nlce_data/celebrity_faces"
SIM_THRESHOLD = 0.45  # cosine similarity threshold for "same identity" (VGGFace2 embeddings)


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")


def ref_path(name: str) -> str:
    return os.path.join(REF_DIR, slug(name) + ".jpg")


def main(args):
    from facenet_pytorch import MTCNN, InceptionResnetV1

    device = args.device
    mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    rows = json.load(open(os.path.join(args.save_dir, "rows.json")))
    target = rows[0]["target"]
    print(f"Target: {target}  |  {len(rows)} rows")

    # ── build reference embeddings (one face crop per person) ──────────────
    @torch.no_grad()
    def embed_reference(name):
        path = ref_path(name)
        if not os.path.exists(path):
            return None
        img = Image.open(path).convert("RGB")
        face = mtcnn(img)
        if face is None:
            return None
        if face.dim() == 4:  # keep_all returns [N,3,160,160] even for single-face detector
            face = face[0]
        emb = resnet(face.unsqueeze(0).to(device))
        return F.normalize(emb, dim=-1)[0]

    ref_cache = {}

    def get_ref(name):
        if name not in ref_cache:
            ref_cache[name] = embed_reference(name)
        return ref_cache[name]

    target_ref = get_ref(target)
    if target_ref is None:
        raise RuntimeError(f"No reference embedding for target '{target}'")

    baseline_dir = os.path.join(args.save_dir, "baseline")
    steered_dir = os.path.join(args.save_dir, "steered")

    @torch.no_grad()
    def detect_and_classify(img_path, target_emb, other_emb):
        """Returns (target_detected, other_detected, target_bbox or None)."""
        img = Image.open(img_path).convert("RGB")
        boxes, _ = mtcnn.detect(img)
        faces = mtcnn(img)
        if faces is None or boxes is None:
            return False, False, None
        if faces.dim() == 3:
            faces = faces.unsqueeze(0)
            boxes = boxes[:1]

        embs = F.normalize(resnet(faces.to(device)), dim=-1)
        target_sims = (embs @ target_emb).cpu().numpy() if target_emb is not None else np.zeros(len(embs))
        other_sims = (embs @ other_emb).cpu().numpy() if other_emb is not None else np.zeros(len(embs))

        target_detected, other_detected, target_bbox = False, False, None
        for i in range(len(embs)):
            t_sim, o_sim = target_sims[i], other_sims[i]
            if t_sim >= SIM_THRESHOLD and t_sim >= o_sim:
                target_detected = True
                target_bbox = boxes[i]
            elif o_sim >= SIM_THRESHOLD and o_sim > t_sim:
                other_detected = True
        return target_detected, other_detected, target_bbox

    import lpips
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    def load_tensor(path, size=(512, 512)):
        img = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
        return t.unsqueeze(0).to(device)

    def masked_lpips(base_path, steer_path, target_bbox, size=(512, 512)):
        bt = load_tensor(base_path, size)
        st = load_tensor(steer_path, size)
        mask = torch.ones(1, 1, size[1], size[0], device=device)
        if target_bbox is not None:
            x1, y1, x2, y2 = target_bbox
            W0, H0 = Image.open(base_path).size
            sx, sy = size[0] / W0, size[1] / H0
            x1, x2 = int(x1 * sx), int(x2 * sx)
            y1, y2 = int(y1 * sy), int(y2 * sy)
            mask[:, :, max(0, y1):min(size[1], y2), max(0, x1):min(size[0], x2)] = 0.0
        with torch.no_grad():
            d = lpips_model(bt * mask, st * mask)
        return float(d.item())

    target_n = {"baseline": 0, "steered": 0, "n": 0}
    other_n = {"baseline": 0, "steered": 0, "n": 0}
    lpips_vals = []
    skipped_no_ref = 0

    for row in tqdm(rows, desc="evaluating"):
        idx, other = row["idx"], row["other"]
        if not other:
            skipped_no_ref += 1
            continue
        other_ref = get_ref(other)
        if other_ref is None:
            skipped_no_ref += 1
            continue

        other_slug = slug(other)
        path_base = os.path.join(baseline_dir, f"{idx}_{other_slug}.png")
        path_steer = os.path.join(steered_dir, f"{idx}_{other_slug}.png")
        if not (os.path.exists(path_base) and os.path.exists(path_steer)):
            continue

        t_det_b, o_det_b, bbox_b = detect_and_classify(path_base, target_ref, other_ref)
        t_det_s, o_det_s, _ = detect_and_classify(path_steer, target_ref, other_ref)

        target_n["n"] += 1
        target_n["baseline"] += int(t_det_b)
        target_n["steered"] += int(t_det_s)
        other_n["n"] += 1
        other_n["baseline"] += int(o_det_b)
        other_n["steered"] += int(o_det_s)

        lpips_vals.append(masked_lpips(path_base, path_steer, bbox_b))

    def pct(c, n):
        return round(100 * c / n, 2) if n else None

    acc_t_base = pct(target_n["baseline"], target_n["n"])
    acc_t_steer = pct(target_n["steered"], target_n["n"])
    acc_r_base = pct(other_n["baseline"], other_n["n"])
    acc_r_steer = pct(other_n["steered"], other_n["n"])

    def harmonic(acc_t, acc_r):
        if acc_t is None or acc_r is None:
            return None
        e, r = 1 - acc_t / 100, acc_r / 100
        return round(2 * e * r / (e + r) * 100, 2) if (e + r) > 0 else 0.0

    h_o_base = harmonic(acc_t_base, acc_r_base)
    h_o_steer = harmonic(acc_t_steer, acc_r_steer)
    lpips_u = round(float(np.mean(lpips_vals)), 4) if lpips_vals else None

    # ── CS: CLIP score (image vs joint-portrait prompt) ─────────────────────
    import clip as openai_clip
    clip_model, clip_preprocess = openai_clip.load("ViT-B/32", device=device)

    def mean_clip_score(img_dir):
        scores = []
        for row in rows:
            if not row["other"] or get_ref(row["other"]) is None:
                continue
            idx, other_slug_ = row["idx"], slug(row["other"])
            path = os.path.join(img_dir, f"{idx}_{other_slug_}.png")
            if not os.path.exists(path):
                continue
            img = clip_preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            text = openai_clip.tokenize([row["prompt"]], truncate=True).to(device)
            with torch.no_grad():
                img_f = F.normalize(clip_model.encode_image(img), dim=-1)
                txt_f = F.normalize(clip_model.encode_text(text), dim=-1)
                scores.append(float((img_f @ txt_f.T).item()) * 100)
        return round(float(np.mean(scores)), 2) if scores else None

    cs_base = mean_clip_score(baseline_dir)
    cs_steer = mean_clip_score(steered_dir)

    # ── KID: cleanfid, baseline (unsteered) vs steered (TCSA) ──────────────
    from cleanfid import fid
    try:
        kid = round(float(fid.compute_kid(baseline_dir, steered_dir, device=device,
                                           use_dataparallel=False)), 4)
    except Exception as e:
        kid = None
        print(f"KID computation failed: {e}")

    results = {
        "target": target, "n_rows": len(rows), "skipped_no_ref": skipped_no_ref,
        "target_erasure": {"n": target_n["n"], "baseline_acc_pct": acc_t_base, "steered_acc_pct": acc_t_steer},
        "other_retention": {"n": other_n["n"], "baseline_acc_pct": acc_r_base, "steered_acc_pct": acc_r_steer},
        "H_o": {"baseline": h_o_base, "steered": h_o_steer},
        "LPIPS_u": lpips_u,
        "CS": {"baseline": cs_base, "steered": cs_steer},
        "KID": kid,
    }

    out_path = os.path.join(args.save_dir, "nlce_celebrity_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Target: {target}  (skipped {skipped_no_ref} rows with no reference photo)")
    print(f"  Acc_t   : {acc_t_steer}%   (baseline {acc_t_base}%)   [lower = better erasure]")
    print(f"  Acc_r   : {acc_r_steer}%   (baseline {acc_r_base}%)   [higher = better retention]")
    print(f"  H_o     : {h_o_steer}%     (baseline {h_o_base}%)     [higher = better balance]")
    print(f"  LPIPS_u : {lpips_u}        [lower = less change outside target region]")
    print(f"  CS      : {cs_steer}       (baseline {cs_base})")
    print(f"  KID     : {kid}")
    print("=" * 70)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    main(args)
