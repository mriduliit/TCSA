"""
merged_unlearning_TECA_SD3.py  —  TECA for Stable Diffusion 3 (MM-DiT)
=======================================================================
Ports the two-space TECA approach to SD3's MM-DiT architecture.

Key architectural differences from SD1.4/SDXL:
  - No UNet: uses SD3Transformer2DModel with JointTransformerBlock layers
  - No cross-attention (attn2): joint attention processes text+image tokens together
  - Three text encoders: CLIP-L, CLIP-G, T5-XXL
  - We steer the IMAGE portion of joint attention output

SPACE 1 — TEXT EMBEDDING:
    Project out concept direction from all three text encoders' embeddings.

SPACE 2 — JOINT ATTENTION STEERING:
    Hook into each JointTransformerBlock's attention output.
    The output has shape [B, N_img + N_txt, D].
    We steer only the image token portion [B, :N_img, D].
    Steering vector v built from paired (concept / no-concept) generations.
"""

import gc
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers import DPMSolverMultistepScheduler, StableDiffusion3Pipeline
from PIL import Image
from tqdm import tqdm as _tqdm

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")
log = logging.getLogger(__name__)

# ── Type aliases ─────────────────────────────────────────────────────────────
# sv[step][layer_idx] = steering vector tensor
SteeringVectors = Dict[int, Dict[int, torch.Tensor]]

# ── Concept vocabularies ──────────────────────────────────────────────────────
CONCEPT_VOCAB: Dict[str, List[str]] = {
    "nudity": [
        "nudity", "nude", "naked", "exposed", "explicit",
        "genitalia", "pornographic", "erotic", "sexual", "nsfw",
        "bare", "undressed", "topless", "bottomless", "intimate",
        "lewd", "indecent", "obscene",
    ],
    "violence": [
        "violence", "violent", "blood", "gore", "killing",
        "murder", "weapon", "brutal", "fight", "attack",
        "injury", "wound", "dead", "corpse", "torture",
    ],
    "hate": [
        "hate", "racism", "racist", "discrimination", "slur",
        "bigotry", "supremacy", "prejudice", "extremist", "hateful",
        "antisemitism", "xenophobia", "homophobia", "misogyny",
    ],
    "self-harm": [
        "suicide", "self-harm", "cutting", "self-injury",
        "overdose", "hanging", "depression", "self-destruction",
        "self-mutilation",
    ],
    "harassment": [
        "harassment", "stalking", "bullying", "cyberbullying", "abuse",
        "threatening", "intimidation", "doxxing",
        "sexual harassment", "coercion", "humiliation", "blackmail",
    ],
    "shocking": [
        "shocking", "disturbing", "grotesque", "disgusting", "horrifying",
        "graphic", "repulsive", "appalling", "nauseating", "revolting",
        "macabre", "visceral", "unsettling",
    ],
    "illegal activity": [
        "illegal activity", "drug dealing", "drug use", "theft", "robbery",
        "criminal activity", "smuggling", "fraud", "hacking",
        "crime scene", "illegal weapons", "contraband",
    ],
}

SAFE_CONCEPT_VOCAB: Dict[str, List[str]] = {
    "nudity": [
        "clothed", "dressed", "modest", "appropriate", "decent",
        "conservative", "covered", "tasteful", "proper", "wholesome",
        "family-friendly", "safe", "non-explicit",
    ],
    "violence": [
        "peaceful", "calm", "gentle", "non-violent", "serene",
        "harmonious", "safe", "friendly", "kind", "cooperative",
    ],
    "hate": [
        "inclusive", "diversity", "equality", "respect", "tolerance",
        "acceptance", "unity", "harmony", "multicultural", "welcoming",
    ],
    "self-harm": [
        "wellness", "health", "happiness", "support", "recovery",
        "hope", "healing", "community", "care", "help",
    ],
    "harassment": [
        "respect", "kindness", "support", "encouragement", "friendship",
        "positive interaction", "empathy", "safety", "dignity",
    ],
    "shocking": [
        "calm", "pleasant", "beautiful", "serene", "wholesome",
        "uplifting", "peaceful", "gentle", "normal", "ordinary",
    ],
    "illegal activity": [
        "lawful", "legal", "legitimate", "civic", "community",
        "everyday life", "normal activity", "peaceful", "authorized",
    ],
}

PAIRED_PROMPTS: Dict[str, List[Tuple[str, str]]] = {
    "nudity": [
        ("a nude woman in a bedroom",            "a woman in a bedroom"),
        ("naked person standing",                "a person standing"),
        ("explicit nude painting",               "a classical painting"),
        ("pornographic image",                   "a portrait"),
        ("erotic photography nude",              "fine art photography"),
        ("nude model artistic",                  "artistic portrait"),
        ("exposed genitalia figure",             "a figure drawing"),
        ("nsfw explicit content",                "a photograph"),
        ("topless woman beach",                  "a woman at the beach"),
        ("unclothed person shower",              "a person in a bathroom"),
    ],
    "violence": [
        ("a violent bloody fight scene",         "two people in a street"),
        ("gore and blood splatter",              "abstract red paint"),
        ("murder victim crime scene",            "an empty room"),
        ("brutal weapon attack",                 "a sports match"),
        ("war violence dead soldiers",           "soldiers in a field"),
        ("knife stabbing blood",                 "a kitchen scene"),
        ("graphic injury wound",                 "a medical illustration"),
        ("violent explosion destruction",        "a building"),
        ("torture chamber scene",                "a dark room"),
        ("mass shooting victims",                "a crowded street"),
    ],
    "hate": [
        ("racist hate symbol graffiti",          "street art graffiti"),
        ("nazi extremist propaganda",            "a historical poster"),
        ("white supremacy rally",                "a public gathering"),
        ("antisemitic slur sign",                "a sign on a street"),
        ("homophobic hate crime",                "two people arguing"),
        ("xenophobic protest banner",            "a protest banner"),
        ("racial discrimination scene",          "people in an office"),
        ("hate group intimidation",              "a group of people"),
        ("misogynist violent imagery",           "a news photograph"),
        ("extremist violent symbol",             "a geometric symbol"),
    ],
    "self-harm": [
        ("person cutting themselves",            "a person sitting alone"),
        ("suicide attempt bridge",               "a person on a bridge"),
        ("self-injury wounds visible",           "a person's arms"),
        ("drug overdose scene",                  "a person lying down"),
        ("depression self-destruction",          "a sad portrait"),
        ("hanging suicide imagery",              "a person in a room"),
        ("self-harm scars visible",              "a close-up portrait"),
        ("suicidal person alone",                "a person at night"),
        ("self-mutilation act",                  "a person's hands"),
        ("medication overdose pills",            "pills on a table"),
    ],
    "harassment": [
        ("a person being violently bullied",     "students in a schoolyard"),
        ("cyberbullying messages on a screen",   "text messages on a phone"),
        ("a stalker following someone",          "two people walking"),
        ("revenge porn screenshot",              "a phone screen"),
        ("threatening letter with images",       "a letter on a desk"),
        ("a person being publicly humiliated",   "a person speaking"),
        ("coercive manipulation scene",          "two people talking"),
        ("doxxing personal information posted",  "information on a screen"),
        ("sexual harassment in a workplace",     "colleagues in an office"),
        ("blackmail message with images",        "a handwritten note"),
    ],
    "shocking": [
        ("a grotesque disturbing crime scene",   "an empty room"),
        ("graphic disgusting body horror",       "a surreal artwork"),
        ("a horrifying accident with victims",   "an empty road"),
        ("revolting nauseating medical imagery", "a medical illustration"),
        ("a macabre display of human remains",   "a historical exhibit"),
        ("deeply disturbing psychological horror","a dramatic portrait"),
        ("repulsive appalling graphic content",  "an abstract painting"),
        ("visceral shock imagery photography",   "a documentary photo"),
        ("extreme unsettling disturbing scene",  "a dimly lit room"),
        ("graphic disturbing image warning",     "a street photograph"),
    ],
    "illegal activity": [
        ("a drug deal in an alley",              "two people in an alley"),
        ("illegal weapons cache basement",       "tools in a basement"),
        ("a robbery in progress at a store",     "people inside a store"),
        ("smuggling contraband border",          "people crossing a border"),
        ("hacker breaking into a computer",      "person at a computer"),
        ("gang illegal transactions street",     "people in a street"),
        ("break-in to a house at night",         "a house at night"),
        ("drug manufacturing in a lab",          "a laboratory"),
        ("fraud counterfeit money production",   "paper and equipment"),
        ("criminal activity on CCTV",            "a CCTV street footage"),
    ],
}


def _paired_prompts(concept: str, n: int) -> List[Tuple[str, str]]:
    if concept not in PAIRED_PROMPTS:
        raise ValueError(f"Unknown concept: {concept}. Supported: {list(PAIRED_PROMPTS)}")
    pairs = list(PAIRED_PROMPTS[concept])
    import random
    while len(pairs) < n:
        pairs += pairs
    return pairs[:n]


# ── Schedule ──────────────────────────────────────────────────────────────────
class Schedule:
    def __init__(self, mode="front", tau_lo=0.0, tau_hi=1.0,
                 tau_relax=0.2, w_min=0.5):
        self.mode = mode
        self.tau_lo = tau_lo
        self.tau_hi = tau_hi
        self.tau_relax = tau_relax
        self.w_min = w_min

    def weight(self, step: int, total: int) -> float:
        t = step / max(total - 1, 1)
        if not (self.tau_lo <= t <= self.tau_hi):
            return 0.0
        if self.mode == "front":
            relax_start = self.tau_hi - self.tau_relax
            if t >= relax_start:
                return self.w_min + (1 - self.w_min) * (
                    (self.tau_hi - t) / max(self.tau_relax, 1e-8))
            return 1.0
        return 1.0

    def summary(self) -> str:
        return (f"Schedule(mode={self.mode}, tau_lo={self.tau_lo}, "
                f"tau_hi={self.tau_hi}, tau_relax={self.tau_relax}, "
                f"w_min={self.w_min})")


# ── SPACE 1: Text embedding orthogonalization ─────────────────────────────────
def _build_text_direction_sd3(concept: str, n_components: int,
                               pipe: StableDiffusion3Pipeline,
                               device: str) -> torch.Tensor:
    """Build concept direction in T5 text embedding space (4096-d)."""
    unsafe_words = CONCEPT_VOCAB.get(concept, [concept])
    safe_words = SAFE_CONCEPT_VOCAB.get(concept, ["safe"])

    def _encode(words):
        vecs = []
        for w in words:
            with torch.no_grad():
                # Use T5 encoder (text_encoder_3) for SD3
                ids = pipe.tokenizer_3(
                    w, return_tensors="pt",
                    padding="max_length",
                    max_length=pipe.tokenizer_3.model_max_length,
                    truncation=True,
                ).input_ids.to(device)
                emb = pipe.text_encoder_3(ids)[0]  # (1, seq, 4096)
                vecs.append(emb.mean(dim=1).squeeze(0))  # (4096,)
        return torch.stack(vecs)  # (N, 4096)

    unsafe_embs = _encode(unsafe_words)
    safe_embs = _encode(safe_words)

    dirs = []
    for _ in range(n_components):
        d = unsafe_embs.mean(0) - safe_embs.mean(0)
        d = d / d.norm().clamp(min=1e-8)
        dirs.append(d)
        # deflate
        unsafe_embs = unsafe_embs - (unsafe_embs @ d).unsqueeze(1) * d
        safe_embs = safe_embs - (safe_embs @ d).unsqueeze(1) * d

    return torch.stack(dirs).to(device)  # (n_components, 4096)


def _orthogonalize_sd3_embeds(
    prompt_embeds: torch.Tensor,      # (B, seq, 4096) — T5
    pooled_embeds: torch.Tensor,      # (B, 2048) — CLIP pooled
    text_dirs: torch.Tensor,          # (n, 4096)
    max_proj_ratio: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project concept direction out of T5 prompt embeddings."""
    out = prompt_embeds.clone()
    for d in text_dirs:
        proj = out @ d  # (B, seq)
        proj_clamped = proj.clamp(max=max_proj_ratio * out.norm(dim=-1))
        out = out - proj_clamped.unsqueeze(-1) * d
    return out, pooled_embeds  # pooled not changed


# ── SPACE 2: Joint-attention steering hook ────────────────────────────────────
class _SD3SteerHook:
    """Hook on SD3 JointTransformerBlock joint attention.

    SD3's joint attention output has shape [B, N_img + N_txt, D].
    We steer only the IMAGE token portion [B, :N_img, D] since that's
    where the concept is rendered. The text portion is left untouched.

    Gate: alpha_k = beta_l * w * tcl * <v, img_token_k>
    """

    def __init__(self, sv: SteeringVectors, beta: float = 2.0,
                 layer_align: Optional[Dict[int, float]] = None,
                 clip: bool = True, align_temp: float = 1.0):
        self.sv = sv or {}
        self.beta = beta
        self.clip = clip
        self._hooks: List = []
        self._step: int = 0
        self._w: float = 0.0
        self._tcl: float = 1.0
        self._layer_idx: int = 0

        # adaptive beta per layer
        self.beta_layer: Dict[int, float] = {}
        if layer_align and layer_align:
            import math
            n = len(layer_align)
            vals = [layer_align[li] for li in sorted(layer_align)]
            max_v = max(vals)
            exps = [math.exp((v - max_v) / max(align_temp, 1e-8)) for v in vals]
            s = sum(exps)
            weights = [(e / s) * n for e in exps]
            self.beta_layer = {
                li: beta * weights[i]
                for i, li in enumerate(sorted(layer_align))
            }
            log.info("Adaptive beta per SD3 layer: %s",
                     [f"{self.beta_layer[li]:.3f}" for li in sorted(self.beta_layer)])
        else:
            log.info("No layer_align — using global beta=%.2f", beta)

    def set_step(self, step: int, w: float, tcl: float = 1.0) -> None:
        self._step = step
        self._w = w
        self._tcl = tcl
        self._layer_idx = 0

    def _lookup(self, li: int) -> Optional[torch.Tensor]:
        try:
            return self.sv[self._step][li]
        except (KeyError, IndexError):
            if not self.sv:
                return None
            closest = min(self.sv.keys(), key=lambda s: abs(s - self._step))
            try:
                return self.sv[closest][li]
            except (KeyError, IndexError):
                return None

    def _hook_fn(self, li: int):
        def fn(module, inp, out):
            w = self._w
            if w <= 0.0:
                return out
            tcl = self._tcl
            v = self._lookup(li)
            if v is None:
                return out

            beta_l = self.beta_layer.get(li, self.beta)
            v = v.to(out.device).to(out.dtype)

            x = out
            # x shape: (B, N_img + N_txt, D)
            # steer only image tokens (first portion)
            # N_img depends on resolution: for 1024px it's (1024/8/2)^2 = 4096
            # we detect N_img as total - N_txt (N_txt is fixed at 256 for T5)
            N_txt = 256  # T5 max_length for SD3
            N_img = x.shape[1] - N_txt

            if N_img <= 0:
                return out

            img_tokens = x[:, :N_img, :]  # (B, N_img, D)

            if x.shape[0] > 1:
                cond = img_tokens[x.shape[0] // 2:].clone()
            else:
                cond = img_tokens.clone()

            dots = cond[0] @ v  # (N_img,)
            alpha = beta_l * w * tcl * dots
            if self.clip:
                alpha = torch.clamp(alpha, min=0.0)

            cond[0] = cond[0] - alpha.unsqueeze(-1) * v.unsqueeze(0)

            x = x.clone()
            if x.shape[0] > 1:
                x[x.shape[0] // 2:, :N_img, :] = cond
            else:
                x[0, :N_img, :] = cond[0]

            return x.to(out.dtype)
        return fn

    def register(self, transformer) -> None:
        """Register hooks on all JointTransformerBlock attention outputs."""
        li = 0
        for name, module in transformer.named_modules():
            if module.__class__.__name__ == "JointTransformerBlock":
                # attn is the joint attention module
                if hasattr(module, "attn"):
                    self._hooks.append(
                        module.attn.register_forward_hook(self._hook_fn(li)))
                    li += 1
        log.info("SD3 joint-attention hooks registered on %d layers", li)

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── CA activation collector for building steering vectors ────────────────────
class _SD3Collector:
    """Collect joint-attention IMAGE token outputs during generation."""

    def __init__(self):
        self._hooks = []
        self.data: Dict[int, Dict[int, List[torch.Tensor]]] = defaultdict(
            lambda: defaultdict(list))
        self._step = 0

    def _hook_fn(self, li: int):
        def fn(module, inp, out):
            N_txt = 256
            N_img = out.shape[1] - N_txt
            if N_img > 0:
                img_mean = out[0, :N_img, :].mean(dim=0).detach().cpu()
                self.data[self._step][li].append(img_mean)
        return fn

    def register(self, transformer) -> None:
        li = 0
        for name, module in transformer.named_modules():
            if module.__class__.__name__ == "JointTransformerBlock":
                if hasattr(module, "attn"):
                    self._hooks.append(
                        module.attn.register_forward_hook(self._hook_fn(li)))
                    li += 1

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── Build steering vectors ────────────────────────────────────────────────────
def build_steering_vectors(
    model_id: str,
    concept: str = "nudity",
    num_pairs: int = 50,
    save_path: Optional[str] = None,
    device: str = "cuda",
    num_steps: int = 28,
    seed: int = 0,
) -> Tuple[SteeringVectors, Dict]:
    """Build per-layer / per-step steering vectors for SD3."""
    if save_path and os.path.exists(save_path):
        log.info("Loading cached steering vectors from %s", save_path)
        data = torch.load(save_path, map_location="cpu", weights_only=False)
        if isinstance(data, dict) and "sv" in data:
            return data["sv"], data.get("layer_align", {})
        return data, {}

    log.info("Building SD3 steering vectors: concept=%s num_pairs=%d",
             concept, num_pairs)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id, torch_dtype=torch.float16,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    pairs = _paired_prompts(concept, num_pairs)
    log.info("Using %d prompt pairs", len(pairs))

    def _collect_mean(prompts):
        collector = _SD3Collector()
        collector.register(pipe.transformer)
        sum_data: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)
        count: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        for i, prompt in enumerate(prompts):
            collector.data = defaultdict(lambda: defaultdict(list))
            collector._step = 0

            def _cb(pipe_, step, timestep, kwargs):
                collector._step = step
                return kwargs

            pipe(
                prompt=prompt,
                num_inference_steps=num_steps,
                generator=torch.Generator(device=device).manual_seed(seed + i),
                callback_on_step_end=_cb,
                output_type="latent",
            )
            for step in collector.data:
                for li in collector.data[step]:
                    vecs = collector.data[step][li]
                    if vecs:
                        v = vecs[0]
                        if li not in sum_data[step]:
                            sum_data[step][li] = v.clone().float()
                            count[step][li] = 1
                        else:
                            sum_data[step][li] += v.float()
                            count[step][li] += 1

        collector.remove()
        means = {}
        for step in sum_data:
            means[step] = {}
            for li in sum_data[step]:
                means[step][li] = sum_data[step][li] / count[step][li]
        return means

    log.info("Collecting POSITIVE activations...")
    pos = _collect_mean([p for p, _ in pairs])
    log.info("Collecting NEGATIVE activations...")
    neg = _collect_mean([n for _, n in pairs])

    sv: SteeringVectors = {}
    for step in pos:
        if step not in neg:
            continue
        sv[step] = {}
        for li in pos[step]:
            if li not in neg[step]:
                continue
            diff = pos[step][li] - neg[step][li]
            sv[step][li] = (diff / diff.norm().clamp(min=1e-8)).cpu()

    # compute layer_align
    layer_align_sum: Dict[int, float] = defaultdict(float)
    layer_align_cnt: Dict[int, int] = defaultdict(int)
    for step in pos:
        if step not in neg:
            continue
        for li in pos[step]:
            if li not in neg[step]:
                continue
            diff = pos[step][li] - neg[step][li]
            layer_align_sum[li] += diff.norm().item()
            layer_align_cnt[li] += 1

    layer_align = {li: layer_align_sum[li] / max(layer_align_cnt[li], 1)
                   for li in layer_align_sum}
    log.info("Layer alignment scores: %s",
             [f"{layer_align[li]:.3f}" for li in sorted(layer_align)])

    log.info("Steering vectors built: %d steps", len(sv))
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save({"sv": sv, "layer_align": layer_align}, save_path)
        log.info("Saved -> %s", save_path)
    return sv, layer_align


# ── SD3 Merged Pipeline ───────────────────────────────────────────────────────
class MergedUnlearningPipelineSD3:
    """TECA applied to Stable Diffusion 3 (MM-DiT architecture)."""

    def __init__(self, pipe, sv, concept="nudity", beta=2.0,
                 n_components=1, max_proj_ratio=0.5,
                 layer_align=None):
        self.pipe = pipe
        self.beta = beta
        self.max_proj_ratio = max_proj_ratio
        self._hook = _SD3SteerHook(sv or {}, beta=beta,
                                   layer_align=layer_align, clip=True)
        self._hook.register(pipe.transformer)
        self.text_dirs = _build_text_direction_sd3(
            concept, n_components, pipe, pipe.device)
        log.info("SD3 text-space direction: %s", tuple(self.text_dirs.shape))

    @classmethod
    def from_pretrained(cls, model_id, sv, device="cuda", concept="nudity",
                        beta=2.0, n_components=1, max_proj_ratio=0.5,
                        layer_align=None):
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id, torch_dtype=torch.float16,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)
        return cls(pipe, sv, concept=concept, beta=beta,
                   n_components=n_components, max_proj_ratio=max_proj_ratio,
                   layer_align=layer_align)

    def _encode_prompt(self, prompt: str):
        """Encode with all three text encoders, apply text orthogonalization."""
        device = self.pipe.device

        # Use pipeline's encode_prompt to handle all 3 encoders
        with torch.no_grad():
            (prompt_embeds, negative_prompt_embeds,
             pooled_embeds, negative_pooled_embeds) = self.pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                prompt_3=prompt,
                negative_prompt="",
                negative_prompt_2="",
                negative_prompt_3="",
                device=device,
            )

        # orthogonalize T5 embeddings (prompt_embeds contains T5 portion)
        prompt_embeds_ortho, pooled_embeds = _orthogonalize_sd3_embeds(
            prompt_embeds, pooled_embeds,
            self.text_dirs, self.max_proj_ratio)

        return (prompt_embeds_ortho, negative_prompt_embeds,
                pooled_embeds, negative_pooled_embeds)

    def generate(self, prompt: str, schedule: Schedule, seed: int = 42,
                 guidance_scale: float = 7.0, num_steps: int = 28,
                 height: int = 1024, width: int = 1024) -> Image.Image:
        device = self.pipe.device
        generator = torch.Generator(device=device).manual_seed(seed)

        # Encode with text orthogonalization
        (prompt_embeds, negative_prompt_embeds,
         pooled_embeds, negative_pooled_embeds) = self._encode_prompt(prompt)

        # compute text concept load (TCL)
        with torch.no_grad():
            tcl = float(torch.cosine_similarity(
                prompt_embeds.mean(dim=1),
                self.text_dirs[0].unsqueeze(0), dim=-1
            ).clamp(0, 1).mean())

        def _cb(pipe_, step, timestep, kwargs):
            w = schedule.weight(step, num_steps)
            self._hook.set_step(step, w, tcl)
            return kwargs

        with torch.no_grad():
            out = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_embeds,
                negative_pooled_prompt_embeds=negative_pooled_embeds,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
                callback_on_step_end=_cb,
            )
        return out.images[0]

    def remove_hooks(self):
        self._hook.remove()


# ── Multi-GPU i2p generation ──────────────────────────────────────────────────
def _i2p_worker(
    rank: int,
    indices: list,
    prompts: list,
    sv_path: str,
    save_dir: str,
    concept: str,
    seed: int,
    model_id: str,
    guidance_scale: float,
    num_steps: int,
    height: int,
    width: int,
    beta: float,
    n_components: int,
    max_proj_ratio: float,
) -> None:
    from tqdm import tqdm
    device = f"cuda:{rank}"
    log.info("[GPU %d] Loading SD3 on %s (%d prompts)", rank, device, len(indices))

    sv, layer_align = build_steering_vectors(
        model_id=model_id, concept=concept,
        save_path=sv_path, device=device)

    wrapper = MergedUnlearningPipelineSD3.from_pretrained(
        model_id, sv, device=device, concept=concept, beta=beta,
        n_components=n_components, max_proj_ratio=max_proj_ratio,
        layer_align=layer_align)

    schedule = Schedule()

    for global_i, prompt in tqdm(
            zip(indices, prompts),
            total=len(indices),
            desc=f"GPU {rank}",
            position=rank,
            miniters=10,
            mininterval=5.0):

        out_path = os.path.join(save_dir, f"{global_i}_0.png")
        if os.path.exists(out_path):
            continue
        if not isinstance(prompt, str) or not prompt.strip():
            continue

        img = wrapper.generate(
            prompt=prompt, schedule=schedule, seed=seed,
            guidance_scale=guidance_scale, num_steps=num_steps,
            height=height, width=width)
        img.save(out_path)

    wrapper.remove_hooks()
    log.info("[GPU %d] done.", rank)


def run_i2p(
    prompts_path: str,
    sv_path: str,
    schedule: Schedule,
    save_dir: str,
    concept: str = "nudity",
    seed: int = 42,
    model_id: str = "stabilityai/stable-diffusion-3-medium-diffusers",
    num_gpus: int = 1,
    guidance_scale: float = 7.0,
    num_steps: int = 28,
    height: int = 1024,
    width: int = 1024,
    beta: float = 2.0,
    n_components: int = 1,
    max_proj_ratio: float = 0.5,
) -> None:
    import pandas as pd
    import torch.multiprocessing as mp

    df = pd.read_csv(prompts_path)
    os.makedirs(save_dir, exist_ok=True)
    all_indices = list(df.index)
    all_prompts = list(df["prompt"].astype(str))
    total = len(all_indices)

    log.info("Prompts: %d | GPUs: %d | Save: %s", total, num_gpus, save_dir)

    if num_gpus == 1:
        _i2p_worker(rank=0, indices=all_indices, prompts=all_prompts,
                    sv_path=sv_path, save_dir=save_dir, concept=concept,
                    seed=seed, model_id=model_id, guidance_scale=guidance_scale,
                    num_steps=num_steps, height=height, width=width,
                    beta=beta, n_components=n_components,
                    max_proj_ratio=max_proj_ratio)
    else:
        chunks_idx = [all_indices[i::num_gpus] for i in range(num_gpus)]
        chunks_pmt = [all_prompts[i::num_gpus] for i in range(num_gpus)]
        mp.set_start_method("spawn", force=True)
        processes = []
        for rank in range(num_gpus):
            p = mp.Process(target=_i2p_worker, kwargs=dict(
                rank=rank, indices=chunks_idx[rank], prompts=chunks_pmt[rank],
                sv_path=sv_path, save_dir=save_dir, concept=concept,
                seed=seed, model_id=model_id, guidance_scale=guidance_scale,
                num_steps=num_steps, height=height, width=width,
                beta=beta, n_components=n_components,
                max_proj_ratio=max_proj_ratio))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        failed = [p.exitcode for p in processes if p.exitcode != 0]
        if failed:
            raise RuntimeError(f"{len(failed)} worker(s) failed: {failed}")

    done = sum(1 for i in all_indices
               if os.path.exists(os.path.join(save_dir, f"{i}_0.png")))
    print(f"\nDone. {done}/{total} images saved to: {save_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def _add_common(sp):
    sp.add_argument("--concept",        default="nudity")
    sp.add_argument("--model_id",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    sp.add_argument("--device",         default="cuda")
    sp.add_argument("--beta",           type=float, default=2.0)
    sp.add_argument("--n_components",   type=int,   default=1)
    sp.add_argument("--max_proj_ratio", type=float, default=0.5)
    sp.add_argument("--guidance",       type=float, default=7.0)
    sp.add_argument("--steps",          type=int,   default=28)
    sp.add_argument("--img_size",       type=int,   default=1024)
    sp.add_argument("--seed",           type=int,   default=42)


def _schedule_from_args(a) -> Schedule:
    return Schedule()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="TECA for Stable Diffusion 3")
    sub = p.add_subparsers(dest="cmd")

    bv = sub.add_parser("build-vectors", help="Build SD3 steering vectors.")
    bv.add_argument("--concept",    default="nudity")
    bv.add_argument("--model_id",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    bv.add_argument("--num_pairs",  type=int,   default=50)
    bv.add_argument("--save_path",  required=True)
    bv.add_argument("--device",     default="cuda")
    bv.add_argument("--steps",      type=int,   default=28)
    bv.add_argument("--seed",       type=int,   default=0)

    i2p = sub.add_parser("i2p", help="Generate images for ASR eval.")
    i2p.add_argument("--prompts",   default="datasets/i2p.csv")
    i2p.add_argument("--sv_path",   required=True)
    i2p.add_argument("--save_dir",  default="results/i2p_eval_sd3")
    i2p.add_argument("--num_gpus",  type=int, default=1)
    _add_common(i2p)

    args = p.parse_args()

    if args.cmd == "build-vectors":
        build_steering_vectors(
            model_id=args.model_id, concept=args.concept,
            num_pairs=args.num_pairs, save_path=args.save_path,
            device=args.device, num_steps=args.steps, seed=args.seed)

    elif args.cmd == "i2p":
        mpr = None if args.max_proj_ratio < 0 else args.max_proj_ratio
        run_i2p(
            prompts_path=args.prompts,
            sv_path=args.sv_path,
            schedule=_schedule_from_args(args),
            save_dir=args.save_dir,
            concept=args.concept,
            seed=args.seed,
            model_id=args.model_id,
            num_gpus=args.num_gpus,
            guidance_scale=args.guidance,
            num_steps=args.steps,
            height=args.img_size,
            width=args.img_size,
            beta=args.beta,
            n_components=args.n_components,
            max_proj_ratio=mpr if mpr else 0.5,
        )
