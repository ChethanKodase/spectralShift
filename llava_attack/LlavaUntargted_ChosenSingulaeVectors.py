


'''

LLaVA-1.5 port of qwen/QwenUntargted_ChosenSingulaeVectors.py. Same attack,
same importance-sampling mechanism, applied to LLaVA-1.5-7b's architecture
instead of Qwen2.5-VL's. Model loading and prompt construction are taken from
llava_attack/llavaInference.py; the module layout (vision tower, language
decoder, multi-modal projector) is taken from llava_attack/model_parameters.txt.

For every targeted linear operator (attention q/k/v -- per attention head --
plus attention output proj, every MLP projection, and both linears of the
vision-to-language projector), this script:

  1. Computes the FULL right-singular-vector basis of its weight
     (torch.linalg.svd(..., full_matrices=True)), matching
     getTopRightSingularVector / getTopRightSingularVectorForVisionQKV /
     getTopRightSingularVectorForLanAttentioHeads from
     qwen/Qwen2p5SpectrumGuidedAttackSameSample.py.

  2. Runs ONE clean forward pass (delta = 0) and `--numRandomVarieties`
     (default 10) forward passes with small random L_inf-epsilon noise
     deltas, and measures, per singular-vector index, how much the
     alignment coefficient shifts between the clean pass and each noisy
     pass:

         diffWeak = sqrt(mean_over_tokens((clean_coeffs - weak_coeffs)^2))

     averaged over the noisy passes -> `weak_mean`, normalized:
     `weak_norm = weak_mean / weak_mean.mean()`.

  3. Selects singular-vector indices via
         indices = torch.where(weak_norm > standardDivCutOff * weak_norm.std())[0]
     (with the same "always keep at least one" fallback as
     QwenUntargted_ChosenSingulaeVectors.py -- if nothing clears the cutoff,
     the single most-sensitive direction is kept).

  4. Minimizes the mean alignment-energy loss between each targeted
     operator's input activations and its chosen singular-vector subspace,
     via Adam over an original-image-space perturbation delta, summed over
     every language / vision / merger target.

ARCHITECTURE DIFFERENCES vs Qwen2.5-VL that this port had to account for
(all derived from llava_attack/model_parameters.txt):

  - Vision tower (CLIP ViT-L/14, "vision_tower.vision_model.encoder.layers.N",
    24 blocks): query/key/value are three SEPARATE nn.Linear modules
    (self_attn.q_proj/k_proj/v_proj, each 1024x1024, with bias), unlike
    Qwen2.5-VL's fused attn.qkv -- so no row-slicing is needed here, they're
    handled exactly like language q/k/v (per-head SVD on each Linear
    directly). Attention output proj is self_attn.out_proj (not attn.proj).
    The vision MLP is a plain 2-layer GELU FFN, mlp.fc1 (1024->4096) /
    mlp.fc2 (4096->1024) -- NOT a 3-way SwiGLU gate/up/down like Qwen's
    vision MLP.

  - Vision-to-language bridge: LLaVA-1.5's multi_modal_projector is a plain
    2-layer GELU MLP of REAL nn.Linear modules -- linear_1 (1024->4096) and
    linear_2 (4096->4096) -- unlike Gemma3's raw (transposed)
    mm_input_projection_weight parameter. Both linears are targeted exactly
    like Qwen's visual.merger.mlp[0]/mlp[2] (no transpose/weight_attr
    special-casing needed, since these are ordinary nn.Linear weights in the
    usual (out_features, in_features) convention).

  - Language decoder ("language_model.model.layers.N", 32 layers): standard
    LLaMA-7B attention -- q_proj/k_proj/v_proj/o_proj are ALL (4096,4096),
    i.e. plain multi-head attention (no GQA, unlike Qwen2.5-VL's and
    Gemma3's language models where num_key_value_heads < num_attention_heads).
    Per-head dimension is still derived directly from each projection's own
    weight shape (out_features // num_heads_for_that_projection) at
    collection time for robustness/consistency with the other ports, even
    though for LLaVA's LLaMA-7B this coincides with hidden_size/num_heads
    (4096/32=128). MLP is standard SwiGLU: gate_proj/up_proj/down_proj
    (11008,4096)/(11008,4096)/(4096,11008).

--standardDivCutOff is recorded in every output filename, same as the Qwen
version.

Example runs:

export CUDA_VISIBLE_DEVICES=2
cd spectralShift/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/LlavaUntargted_ChosenSingulaeVectors.py --attck_type ImpSamp --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --standardDivCutOff 5
done

# Restrict to a subset of layers / fewer random noise varieties if desired:
python llava_attack/LlavaUntargted_ChosenSingulaeVectors.py --attck_type saa_loop --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 --attackSample 1 --standardDivCutOff 3 --numRandomVarieties 10 --chosenLanLayers 2 --chosenVisLayers 0 1 2 4 5 6 7 8 9 14 24

'''

#!/usr/bin/env python
import os
import re
import argparse
import random
from collections import OrderedDict
import numpy as np

# Determinism env vars should be set before torch is heavily used
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import (
    LlavaForConditionalGeneration,
    CLIPImageProcessor,
    LlamaTokenizer,
)


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(42)

try:
    torch.use_deterministic_algorithms(True)
except Exception as e:
    print(f"[WARN] Could not enable deterministic algorithms: {e}")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

criterion = nn.MSELoss()


# ----------------------------
# Losses/utilities kept for compatibility
# ----------------------------
def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()


def wasserstein_distance(tensor_a, tensor_b):
    tensor_a_flat = torch.flatten(tensor_a)
    tensor_b_flat = torch.flatten(tensor_b)
    tensor_a_sorted, _ = torch.sort(tensor_a_flat)
    tensor_b_sorted, _ = torch.sort(tensor_b_flat)
    return torch.mean(torch.abs(tensor_a_sorted - tensor_b_sorted))


def get_oa_l2(outputs, outputsN):
    return criterion(outputs.logits, outputsN.logits)


# ----------------------------
# PIL/image tensor helpers
# ----------------------------
def pil_to_tensor01(pil_img: Image.Image) -> torch.Tensor:
    """PIL RGB -> torch float tensor in [0,1], shape (1,3,H,W)."""
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor01_to_pil(t01: torch.Tensor) -> Image.Image:
    """torch tensor [0,1], shape (1,3,H,W) or (3,H,W) -> PIL RGB."""
    if t01.dim() == 4:
        t01 = t01[0]
    t01 = t01.detach().cpu().clamp(0, 1)
    arr = (t01.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ----------------------------
# LLaVA (CLIP) differentiable preprocessing
# Reuses the same generic resize + center-crop + normalize pipeline the
# Gemma3 port uses (gemma_attack/gemma3Inference.py) -- it already reads
# crop_size / size generically from the HF image processor, and
# CLIPImageProcessor exposes those in the exact same shapes (crop_size as
# {"height":.., "width":..}, size as {"shortest_edge":..}), so no
# LLaVA-specific changes were needed here.
# ----------------------------
def _get_target_hw(image_processor):
    ip = image_processor
    target_h = target_w = None

    crop = getattr(ip, "crop_size", None)
    if isinstance(crop, dict):
        target_h = crop.get("height", None)
        target_w = crop.get("width", None)
    elif isinstance(crop, int):
        target_h = target_w = crop

    if target_h is None or target_w is None:
        size = getattr(ip, "size", None)
        if isinstance(size, dict):
            if "height" in size and "width" in size:
                target_h = size["height"]
                target_w = size["width"]
            elif "shortest_edge" in size:
                target_h = target_w = size["shortest_edge"]
        elif isinstance(size, int):
            target_h = target_w = size

    if target_h is None or target_w is None:
        # fallback for many CLIP-style vision configs
        target_h = target_w = 336

    return int(target_h), int(target_w)


def resize_keep_aspect_center_crop(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    _, _, H, W = x.shape
    scale = max(target_h / H, target_w / W)
    newH = int(round(H * scale))
    newW = int(round(W * scale))

    x_resized = F.interpolate(x, size=(newH, newW), mode="bilinear", align_corners=False)

    top = max((newH - target_h) // 2, 0)
    left = max((newW - target_w) // 2, 0)
    x_crop = x_resized[:, :, top:top + target_h, left:left + target_w]

    # pad if needed (unlikely)
    pad_h = target_h - x_crop.shape[2]
    pad_w = target_w - x_crop.shape[3]
    if pad_h > 0 or pad_w > 0:
        x_crop = F.pad(x_crop, (0, max(pad_w, 0), 0, max(pad_h, 0)))

    return x_crop


def normalize_like_processor(x01: torch.Tensor, image_processor) -> torch.Tensor:
    mean = torch.tensor(image_processor.image_mean, dtype=x01.dtype, device=x01.device).view(1, 3, 1, 1)
    std = torch.tensor(image_processor.image_std, dtype=x01.dtype, device=x01.device).view(1, 3, 1, 1)
    return (x01 - mean) / std


def llava_preprocess_differentiable(x01: torch.Tensor, image_processor) -> torch.Tensor:
    th, tw = _get_target_hw(image_processor)
    x = resize_keep_aspect_center_crop(x01, th, tw)
    x = normalize_like_processor(x, image_processor)
    return x


# ----------------------------
# Build template inputs once
# (prompt format ported verbatim from llava_attack/llavaInference.py --
#  LLaVA-1.5 uses a fixed "USER: <image>\n...\nASSISTANT:" template, not a
#  chat-template call like Qwen/Gemma3.)
#
# NOTE: text is tokenized directly via the raw LlamaTokenizer, exactly like
# llava_attack/llava_attack_imagenet_SSGRA.py's build_template_inputs(). No
# LlavaProcessor is constructed anywhere in this file (importing/constructing
# it triggers a `transformers.utils.versions.require_version` check for the
# `regex` package that isn't registered with proper metadata in the llava15
# conda env -- see the ImportError/PackageNotFoundError this used to raise).
# Keeping tokenizer and image_processor fully separate, as the known-working
# SSGRA reference script does, avoids that code path entirely.
# ----------------------------
def build_template_inputs(tokenizer, question: str, device):
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    enc = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


# ----------------------------
# Generation helper
# (ported from llava_attack/llava_attack_imagenet_SSGRA.py -- decodes via the
#  raw tokenizer, not a LlavaProcessor.)
# ----------------------------
def run_generation_with_pixel_values(model, tokenizer, template_inputs, pixel_values, max_new_tokens=128):
    model.eval()
    inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
    inputs["pixel_values"] = pixel_values

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_ids = inputs["input_ids"]
    gen_only = out_ids[:, input_ids.shape[1]:]
    return tokenizer.decode(gen_only[0], skip_special_tokens=True)


# ============================================================
# Chosen-singular-vector selected module attack for LLaVA-1.5
# ============================================================
layer_inputs = {}


def make_pre_hook(name):
    def hook(module, inputs):
        layer_inputs[name] = inputs[0]
    return hook


def extract_language_layer_idx(name: str):
    patterns = [
        r"language_model\.model\.layers\.(\d+)\.",
        r"language_model\.layers\.(\d+)\.",
        r"model\.layers\.(\d+)\.",
    ]
    for p in patterns:
        m = re.search(p, name)
        if m is not None:
            return int(m.group(1))
    return None


def extract_vision_layer_idx(name: str):
    patterns = [
        r"visual\.blocks\.(\d+)\.",
        r"visual\.merger\..*layers\.(\d+)\.",
        r"vision_tower\.vision_model\.encoder\.layers\.(\d+)\.",
        r"vision_model\.encoder\.layers\.(\d+)\.",
    ]
    for p in patterns:
        m = re.search(p, name)
        if m is not None:
            return int(m.group(1))
    return None


# LLaVA-1.5's vision-to-language bridge (multi_modal_projector) is a plain
# 2-layer GELU MLP of REAL nn.Linear modules -- linear_1 / linear_2 -- so,
# like Qwen's visual.merger.mlp.0/2, both are matched directly by name
# inside the same named_modules() scan collect_target_specs() already runs
# (no separate raw-parameter special-casing needed, unlike Gemma3's
# multi_modal_projector.mm_input_projection_weight).
MERGER_NAME_PATTERN = re.compile(r"multi_modal_projector\.linear_(1|2)$")
MERGER_LABEL = "Vis-to-lan proj"


def is_merger_target(name: str) -> bool:
    return bool(MERGER_NAME_PATTERN.search(name))


def _resolve_vision_config(model):
    cfg = model.config
    return getattr(cfg, "vision_config", cfg)


def _resolve_text_config(model):
    cfg = model.config
    return getattr(cfg, "text_config", cfg)


def get_vision_head_config(model):
    vision_cfg = _resolve_vision_config(model)
    num_heads = int(getattr(vision_cfg, "num_attention_heads", getattr(vision_cfg, "num_heads", 16)))
    return num_heads


def get_language_head_config(model):
    text_cfg = _resolve_text_config(model)
    num_heads = int(getattr(text_cfg, "num_attention_heads"))
    num_kv_heads = int(getattr(text_cfg, "num_key_value_heads", num_heads))
    return num_heads, num_kv_heads


# ----------------------------
# Target-module discovery (every language layer, every vision block, merger)
# ----------------------------
def collect_target_specs(
    model,
    chosen_lan_layers=None,
    chosen_vis_layers=None,
    include_merger=True,
    vis_num_heads=16,
    lan_num_heads=32,
    lan_num_kv_heads=32,
):
    """
    LLaVA-1.5 analogue of the Qwen version's collect_target_specs. Attention
    q/k/v projections are marked is_head=True with the per-head geometry
    needed to reproduce getTopRightSingularVectorForLanAttentioHeads-style
    per-head SVDs -- num_heads comes from config, and d_head/d_in are
    derived directly from each projection's own weight shape (kept generic
    rather than assuming hidden_size/num_heads, for consistency with the
    Gemma3 port, even though LLaVA's LLaMA-7B language model happens to
    have no head_dim/hidden_size decoupling and no GQA).
    """
    chosen_lan_layers_set = None if chosen_lan_layers is None else set(chosen_lan_layers)
    chosen_vis_layers_set = None if chosen_vis_layers is None else set(chosen_vis_layers)

    specs = []
    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue
        if not torch.is_tensor(module.weight):
            continue
        if module.weight.ndim != 2:
            continue

        # ---------------- language decoder layers ----------------
        lan_idx = extract_language_layer_idx(name)
        if lan_idx is not None and "vision_tower" not in name:
            if chosen_lan_layers_set is not None and lan_idx not in chosen_lan_layers_set:
                continue

            base = dict(hook_name=name, module=module, kind="language", layer_idx=lan_idx, weight_slice=None)

            if name.endswith(".self_attn.q_proj"):
                d_out, d_in = module.weight.shape
                specs.append(dict(base, name=name, sub_kind="query proj (lan)", is_head=True,
                                   num_heads=lan_num_heads, d_head=d_out // lan_num_heads, d_in=d_in))
            elif name.endswith(".self_attn.k_proj"):
                d_out, d_in = module.weight.shape
                specs.append(dict(base, name=name, sub_kind="key proj (lan)", is_head=True,
                                   num_heads=lan_num_kv_heads, d_head=d_out // lan_num_kv_heads, d_in=d_in))
            elif name.endswith(".self_attn.v_proj"):
                d_out, d_in = module.weight.shape
                specs.append(dict(base, name=name, sub_kind="value proj (lan)", is_head=True,
                                   num_heads=lan_num_kv_heads, d_head=d_out // lan_num_kv_heads, d_in=d_in))
            elif name.endswith(".self_attn.o_proj"):
                specs.append(dict(base, name=name, sub_kind="att output proj (lan)", is_head=False,
                                   num_heads=None, d_head=None, d_in=None))
            elif name.endswith(".mlp.gate_proj"):
                specs.append(dict(base, name=name, sub_kind="MLP gate proj", is_head=False,
                                   num_heads=None, d_head=None, d_in=None))
            elif name.endswith(".mlp.up_proj"):
                specs.append(dict(base, name=name, sub_kind="MLP up proj", is_head=False,
                                   num_heads=None, d_head=None, d_in=None))
            elif name.endswith(".mlp.down_proj"):
                specs.append(dict(base, name=name, sub_kind="MLP down proj", is_head=False,
                                   num_heads=None, d_head=None, d_in=None))
            continue

        # ---------------- vision transformer blocks (CLIP ViT-L/14) ----------------
        vis_idx = extract_vision_layer_idx(name)
        if vis_idx is not None and "vision_tower" in name:
            if chosen_vis_layers_set is not None and vis_idx not in chosen_vis_layers_set:
                continue

            base = dict(hook_name=name, module=module, kind="vision", layer_idx=vis_idx, weight_slice=None)

            if name.endswith(".self_attn.q_proj"):
                d_out, d_in = module.weight.shape
                specs.append(dict(base, name=name, sub_kind="query proj (vis)", is_head=True,
                                   num_heads=vis_num_heads, d_head=d_out // vis_num_heads, d_in=d_in))
            elif name.endswith(".self_attn.k_proj"):
                d_out, d_in = module.weight.shape
                specs.append(dict(base, name=name, sub_kind="key proj (vis)", is_head=True,
                                   num_heads=vis_num_heads, d_head=d_out // vis_num_heads, d_in=d_in))
            elif name.endswith(".self_attn.v_proj"):
                d_out, d_in = module.weight.shape
                specs.append(dict(base, name=name, sub_kind="value proj (vis)", is_head=True,
                                   num_heads=vis_num_heads, d_head=d_out // vis_num_heads, d_in=d_in))
            elif name.endswith(".self_attn.out_proj"):
                specs.append(dict(base, name=name, sub_kind="att output proj (vis)", is_head=False,
                                   num_heads=None, d_head=None, d_in=None))
            elif name.endswith(".mlp.fc1"):
                specs.append(dict(base, name=name, sub_kind="MLP fc1 (vis)", is_head=False,
                                   num_heads=None, d_head=None, d_in=None))
            elif name.endswith(".mlp.fc2"):
                specs.append(dict(base, name=name, sub_kind="MLP fc2 (vis)", is_head=False,
                                   num_heads=None, d_head=None, d_in=None))
            continue

        # ---------------- vision-to-language bridge ----------------
        if include_merger and is_merger_target(name):
            specs.append(dict(hook_name=name, module=module, kind="merger", layer_idx=None,
                               name=name, sub_kind=MERGER_LABEL, is_head=False,
                               weight_slice=None, num_heads=None, d_head=None, d_in=None))

    return specs


def register_all_hooks(specs):
    handles = []
    seen_hook_names = set()
    for spec in specs:
        hook_name = spec["hook_name"]
        if hook_name in seen_hook_names:
            continue
        seen_hook_names.add(hook_name)
        h = spec["module"].register_forward_pre_hook(make_pre_hook(hook_name))
        handles.append(h)
    return handles


# ----------------------------
# Full singular-vector basis (matches getTopRightSingularVector /
# getTopRightSingularVectorForVisionQKV / getTopRightSingularVectorForLanAttentioHeads)
# ----------------------------
def compute_full_vh_for_spec(spec):
    with torch.no_grad():
        W = spec["module"].weight.detach().to(torch.float32)
        if spec["weight_slice"] is not None:
            start, end = spec["weight_slice"]
            W = W[start:end, :]

        if spec["is_head"]:
            num_heads = spec["num_heads"]
            d_head = spec["d_head"]
            d_in = spec["d_in"]
            W = W.reshape(num_heads, d_head, d_in)
            vh_list = []
            for h in range(num_heads):
                Vh_h = torch.linalg.svd(W[h], full_matrices=True)[2]  # (d_in, d_in)
                vh_list.append(Vh_h)
            Vh = torch.stack(vh_list, dim=0)  # (num_heads, d_in, d_in)
        else:
            Vh = torch.linalg.svd(W, full_matrices=True)[2]  # (n, n)
    return Vh


def _flatten_tokens(H):
    if isinstance(H, (tuple, list)):
        H = H[0]
    if H.dim() == 3:
        H = H.reshape(-1, H.shape[-1])
    elif H.dim() == 2:
        pass
    else:
        H = H.view(-1, H.shape[-1])
    return H


def _coeffs_against_full_vh(H, Vh, is_head):
    """
    Mirrors getMeanAlignmentWithTopRightSingularVector (is_head=False) /
    getMeanAlignmentWithAttentionHeadTopRightSingularVector (is_head=True),
    returning the raw per-token, per-singular-vector coefficients.
    """
    H = _flatten_tokens(H)
    H_hat = F.normalize(H, dim=1)
    if is_head:
        V_hat = F.normalize(Vh, dim=2)                      # (heads, n, n)
        coeffs = torch.einsum('nd,hkd->hnk', H_hat, V_hat)  # (heads, N, n)
    else:
        V_hat = F.normalize(Vh, dim=1)                      # (n, n)
        coeffs = H_hat @ V_hat.T                            # (N, n)
    return coeffs


def getMeanAlignmentLossWithChosenSubspace(InputToLayer, chosenRightSingularVectors):
    """
    Same alignment-energy loss as the Qwen version's
    getMeanAlignmentLossWithChosenSubspace.
    """
    H = _flatten_tokens(InputToLayer)
    V = chosenRightSingularVectors.to(device=H.device, dtype=H.dtype)

    H_hat = F.normalize(H, dim=1)
    V_hat = F.normalize(V, dim=1)

    coeffs = H_hat @ V_hat.T
    per_token_energy = (coeffs ** 2).sum(dim=1)
    loss = ((1.0 - per_token_energy) ** 2).mean()
    return loss


# ----------------------------
# Importance sampling: pick singular-vector indices via the same
# clean-vs-random-noise sensitivity criterion as
# Qwen2p5SpectrumGuidedAttackSameSample.py's
#   indices = torch.where(weak_norm > standardDivCutOff * weak_norm.std())[0]
# ----------------------------
def run_importance_sampling(
    model,
    image_processor,
    template_inputs,
    x_orig01,
    epsilon,
    device,
    target_specs,
    standardDivCutOff: float,
    numRandomVarieties: int,
):
    unique_hook_names = sorted(set(s["hook_name"] for s in target_specs))

    def run_one_pass(delta):
        layer_inputs.clear()
        x_p01 = (x_orig01 + delta).clamp(0.0, 1.0)
        x_p01 = torch.max(torch.min(x_p01, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

        pv = llava_preprocess_differentiable(x_p01, image_processor)
        inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
        inputs["pixel_values"] = pv
        inputs["labels"] = template_inputs["input_ids"]
        inputs["use_cache"] = False

        with torch.no_grad():
            model(**inputs, output_hidden_states=False, return_dict=True)

        snapshot = {}
        for hn in unique_hook_names:
            if hn in layer_inputs:
                H = layer_inputs[hn]
                if isinstance(H, (tuple, list)):
                    H = H[0]
                # Keep snapshots on CPU between passes so the 11 forward
                # passes (clean + numRandomVarieties noisy) don't have to
                # hold every hooked activation on the GPU simultaneously.
                snapshot[hn] = H.detach().to(torch.float32).cpu()
        return snapshot

    print(f"\n[INFO] Importance sampling: 1 clean pass + {numRandomVarieties} weak-noise "
          f"passes (epsilon={epsilon}) over {len(unique_hook_names)} hooked modules ...")

    clean_snapshot = run_one_pass(torch.zeros_like(x_orig01))
    if device.type == "cuda":
        torch.cuda.empty_cache()

    weak_snapshots = []
    for v in range(numRandomVarieties):
        weak_delta = torch.randn_like(x_orig01) * epsilon
        weak_snapshots.append(run_one_pass(weak_delta))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("[INFO] Forward passes complete. Selecting importance-sampled singular-vector subspaces ...")
    print("\n========== CHOSEN SINGULAR-VECTOR SUBSPACES (ALL LAYER TYPES, ALL BLOCKS) ==========")

    num_kept = 0
    for spec_i, spec in enumerate(target_specs):
        hn = spec["hook_name"]
        if hn not in clean_snapshot:
            spec["target_vectors"] = None
            continue

        Vh = compute_full_vh_for_spec(spec).to(device)
        n = Vh.shape[-1]

        H_clean = clean_snapshot[hn].to(device)
        orig_coeffs = _coeffs_against_full_vh(H_clean, Vh, spec["is_head"])

        diffs = []
        for v in range(numRandomVarieties):
            H_weak = weak_snapshots[v][hn].to(device)
            weak_coeffs = _coeffs_against_full_vh(H_weak, Vh, spec["is_head"])
            token_dim = 0 if weak_coeffs.dim() == 2 else 1
            diff_v = (orig_coeffs - weak_coeffs).pow(2).mean(dim=token_dim).sqrt().reshape(-1)
            diffs.append(diff_v)
            del H_weak, weak_coeffs

        diffs = torch.stack(diffs, dim=0)             # (numRandomVarieties, flat_len)
        weak_mean = diffs.mean(dim=0)                 # (flat_len,)
        weak_mean_avg = weak_mean.mean()
        weak_norm = weak_mean / (weak_mean_avg + 1e-12)
        cutoff = standardDivCutOff * weak_norm.std()

        chosen_flat_idx = torch.where(weak_norm > cutoff)[0]
        if chosen_flat_idx.numel() == 0:
            # Safety fallback: always keep at least one (the most sensitive) direction.
            chosen_flat_idx = torch.argmax(weak_norm).view(1)

        if spec["is_head"]:
            head_idx = torch.div(chosen_flat_idx, n, rounding_mode="floor")
            row_idx = chosen_flat_idx % n
            target_vectors = Vh[head_idx, row_idx, :].contiguous()
        else:
            target_vectors = Vh[chosen_flat_idx, :].contiguous()

        spec["target_vectors"] = target_vectors.detach().clone()
        spec["num_chosen"] = int(chosen_flat_idx.numel())
        num_kept += 1

        layer_str = f"{spec['layer_idx']:3d}" if spec["layer_idx"] is not None else "N/A"
        head_tag = f"{Vh.shape[0]}heads x {n}" if spec["is_head"] else f"{n}"
        print(
            f"{spec['kind']:8s} | layer={layer_str} | {spec['sub_kind']:24s} | {hn} | "
            f"full_basis={head_tag} | weak_norm.std={float(weak_norm.std().item()):.6f} "
            f"| chosen={spec['num_chosen']}"
        )

        del Vh, H_clean, orig_coeffs, diffs
        if device.type == "cuda" and spec_i % 8 == 0:
            torch.cuda.empty_cache()

    print("=====================================================================================")
    print(f"Total targets with a chosen subspace: {num_kept} / {len(target_specs)}")
    print("=====================================================================================\n")

    del clean_snapshot, weak_snapshots
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return [s for s in target_specs if s.get("target_vectors") is not None]


def aggregated_target_subspace_loss(target_specs, device):
    lang_losses = []
    vis_losses = []
    merger_losses = []

    for spec in target_specs:
        hn = spec["hook_name"]
        if hn not in layer_inputs:
            continue

        H = layer_inputs[hn]
        loss_i = getMeanAlignmentLossWithChosenSubspace(H, spec["target_vectors"])

        if spec["kind"] == "language":
            lang_losses.append(loss_i)
        elif spec["kind"] == "vision":
            vis_losses.append(loss_i)
        elif spec["kind"] == "merger":
            merger_losses.append(loss_i)

    language_loss = torch.tensor(0.0, device=device)
    vision_loss = torch.tensor(0.0, device=device)
    merger_loss = torch.tensor(0.0, device=device)

    if len(lang_losses) > 0:
        language_loss = torch.stack(lang_losses).mean()

    if len(vis_losses) > 0:
        vision_loss = torch.stack(vis_losses).mean()

    if len(merger_losses) > 0:
        merger_loss = torch.stack(merger_losses).mean()

    total_used = len(lang_losses) + len(vis_losses) + len(merger_losses)
    return language_loss, vision_loss, merger_loss, total_used


# ----------------------------
# ORIGINAL-SPACE Adam attack
# ----------------------------
def adam_attack_original_space(
    model,
    image_processor,
    template_inputs,
    x_orig01,
    attck_type: str,
    num_steps: int,
    lr: float,
    epsilon: float,
    device,
    save_conv_path: str,
    AttackStartLayer: int,
    numLayerstAtAtime: int,
    standardDivCutOff: float,
    numRandomVarieties: int,
    chosenLanLayers=None,
    chosenVisLayers=None,
    includeMerger=True,
):
    x_orig01 = x_orig01.detach().to(device)

    delta = 0.001 * torch.randn_like(x_orig01, device=device)
    delta.requires_grad_(True)

    opt = torch.optim.Adam([delta], lr=lr)

    losses_list = []
    best_loss = 1e20
    best_delta = delta.detach().clone()

    model.eval()

    vis_num_heads = get_vision_head_config(model)
    lan_num_heads, lan_num_kv_heads = get_language_head_config(model)

    target_specs = collect_target_specs(
        model,
        chosen_lan_layers=chosenLanLayers,
        chosen_vis_layers=chosenVisLayers,
        include_merger=includeMerger,
        vis_num_heads=vis_num_heads,
        lan_num_heads=lan_num_heads,
        lan_num_kv_heads=lan_num_kv_heads,
    )

    if len(target_specs) == 0:
        raise RuntimeError(
            "No target modules found. Check --chosenLanLayers and --chosenVisLayers."
        )

    hook_handles = register_all_hooks(target_specs)

    target_specs = run_importance_sampling(
        model,
        image_processor,
        template_inputs,
        x_orig01,
        epsilon,
        device,
        target_specs,
        standardDivCutOff=standardDivCutOff,
        numRandomVarieties=numRandomVarieties,
    )

    if len(target_specs) == 0:
        for h in hook_handles:
            h.remove()
        raise RuntimeError("Importance sampling selected no usable target subspaces.")

    adv_inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
    adv_inputs["labels"] = template_inputs["input_ids"]
    adv_inputs["use_cache"] = False

    for step in range(num_steps):
        layer_inputs.clear()

        x_adv01 = (x_orig01 + delta).clamp(0.0, 1.0)
        x_adv01 = torch.max(torch.min(x_adv01, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

        pv_adv = llava_preprocess_differentiable(x_adv01, image_processor)
        adv_inputs["pixel_values"] = pv_adv

        outputs = model(**adv_inputs, output_hidden_states=False, return_dict=True)

        language_loss, vision_loss, merger_loss, total_used = aggregated_target_subspace_loss(target_specs, device=device)

        if total_used == 0:
            raise RuntimeError("No hooked target modules were used in the forward pass.")

        loss = language_loss + vision_loss + merger_loss
        attack_loss = loss

        opt.zero_grad(set_to_none=True)
        attack_loss.backward()
        opt.step()

        with torch.no_grad():
            delta.data.clamp_(-epsilon, epsilon)

        lv = float(loss.item())
        losses_list.append(lv)

        if (step + 1) % 10 == 0 or step == 0:
            print(
                f"[step {step+1}/{num_steps}] "
                f"total_loss={lv:.6f} "
                f"language_loss={float(language_loss.item()):.6f} "
                f"vision_loss={float(vision_loss.item()):.6f} "
                f"merger_loss={float(merger_loss.item()):.6f} "
                f"used_modules={total_used}"
            )

        if lv < best_loss:
            best_loss = lv
            best_delta = delta.detach().clone()
            np.save(save_conv_path, np.array(losses_list, dtype=np.float32))

        del outputs, loss, attack_loss, pv_adv
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for h in hook_handles:
        h.remove()

    with torch.no_grad():
        x_adv01_final = (x_orig01 + best_delta).clamp(0.0, 1.0)
        x_adv01_final = torch.max(torch.min(x_adv01_final, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

    return x_adv01_final, best_delta


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="LLaVA-1.5 ORIGINAL-image-space adversarial attack, importance-sampled singular-vector subspaces, ALL layer types, ALL blocks"
    )
    parser.add_argument("--attck_type", type=str, default="grill_l2", help="kept for compatibility")
    parser.add_argument("--desired_norm_l_inf", type=float, default=0.03, help="epsilon L_inf in ORIGINAL pixel space [0..1]")
    parser.add_argument("--learningRate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--num_steps", type=int, default=None, help="Number of Adam steps")
    parser.add_argument("--numSteps", type=int, default=None, help="Number of Adam steps, old-style name")
    parser.add_argument("--attackSample", type=str, default="1", help="which sample")
    parser.add_argument("--AttackStartLayer", type=int, default=0, help="kept for compatibility")
    parser.add_argument("--numLayerstAtAtime", type=int, default=2, help="kept for compatibility")

    parser.add_argument(
        "--standardDivCutOff",
        type=float,
        default=3.0,
        help=(
            "Singular-vector indices are kept where "
            "weak_norm > standardDivCutOff * weak_norm.std(), exactly as in "
            "Qwen2p5SpectrumGuidedAttackSameSample.py, and the resulting "
            "importance-sampled singular vectors span the target subspace."
        ),
    )
    parser.add_argument(
        "--numRandomVarieties",
        type=int,
        default=10,
        help="Number of random weak-noise passes used to estimate weak_norm.",
    )

    parser.add_argument("--AlignLayer", type=int, default=None, help="old-style arg; used as chosenLanLayers if --chosenLanLayers is omitted")

    parser.add_argument(
        "--chosenLanLayers",
        type=int,
        nargs="+",
        default=None,
        help="Space-separated language layer indices to attack. Omit for ALL language layers (0-31).",
    )
    parser.add_argument(
        "--chosenVisLayers",
        type=int,
        nargs="+",
        default=None,
        help="Space-separated vision layer indices to attack. Omit for ALL vision blocks (0-23).",
    )
    parser.add_argument(
        "--includeMerger",
        dest="includeMerger",
        action="store_true",
        default=True,
        help="Include the vision-to-language multi_modal_projector targets (default: on).",
    )
    parser.add_argument(
        "--no-includeMerger",
        dest="includeMerger",
        action="store_false",
        help="Exclude the vision-to-language multi_modal_projector targets.",
    )

    args = parser.parse_args()

    attck_type = args.attck_type
    epsilon = float(args.desired_norm_l_inf)
    lr = float(args.learningRate)
    num_steps = args.num_steps if args.num_steps is not None else args.numSteps
    if num_steps is None:
        num_steps = 2000
    num_steps = int(num_steps)

    attackSample = str(args.attackSample)
    AttackStartLayer = int(args.AttackStartLayer)
    numLayerstAtAtime = int(args.numLayerstAtAtime)
    standardDivCutOff = float(args.standardDivCutOff)
    numRandomVarieties = int(args.numRandomVarieties)
    includeMerger = bool(args.includeMerger)

    chosenLanLayers = args.chosenLanLayers
    if chosenLanLayers is None and args.AlignLayer is not None:
        chosenLanLayers = [int(args.AlignLayer)]

    chosenVisLayers = args.chosenVisLayers

    lanLayersTag = "ALL" if chosenLanLayers is None else "-".join(map(str, chosenLanLayers))
    visLayersTag = "ALL" if chosenVisLayers is None else "-".join(map(str, chosenVisLayers))

    # Preserve llava_attack paths (see llava_attack/llavaInference.py).
    MODEL_PATH = "/home/luser/LLaVA/llava-1.5-7b-hf"
    IMAGE_PATH = f"../interpretAttacks/llava_attack/dataSamplesForQuant/{attackSample}.JPEG"
    QUESTION = "What is shown in this image?"
    MAX_NEW_TOKENS = 128

    os.makedirs("llava_attack/outputsStorageImagenet", exist_ok=True)
    os.makedirs(f"llava_attack/outputsStorageImagenet/advOutputs/{attackSample}", exist_ok=True)
    os.makedirs(f"llava_attack/outputsStorageImagenet/convergence/{attackSample}", exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"device={device}, dtype={dtype}")

    print("Loading tokenizer + image_processor...")
    # Kept as fully separate objects -- no LlavaProcessor is constructed
    # anywhere in this script, matching the known-working reference
    # llava_attack/llava_attack_imagenet_SSGRA.py. Constructing a
    # LlavaProcessor triggers a transformers.utils.versions.require_version
    # check for the `regex` package that isn't registered with proper
    # metadata in the llava15 conda env, which raised
    # `importlib.metadata.PackageNotFoundError: No package metadata was
    # found for regex`. Text is tokenized directly via the raw tokenizer,
    # and pixel_values come from the differentiable preprocessing pipeline
    # driven by the raw image_processor.
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    image_processor = CLIPImageProcessor.from_pretrained(MODEL_PATH)

    print("Loading model...")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if device.type == "cpu":
        model = model.to(device)
    model.config.use_cache = False

    print("\n[INFO] LLaVA-1.5 ALL-layer, ALL-block, CHOSEN-singular-vector variant.")
    print("[INFO] For every q/k/v/o and MLP projection in every vision block, every")
    print("[INFO] language layer, and both multi_modal_projector linears, singular-vector")
    print("[INFO] indices are importance-sampled via the clean-vs-random-noise")
    print("[INFO] sensitivity criterion from Qwen2p5SpectrumGuidedAttackSameSample.py,")
    print("[INFO] then used to span the alignment target subspace.")
    print(f"[INFO] chosenLanLayers={chosenLanLayers if chosenLanLayers is not None else 'ALL (0-31)'}")
    print(f"[INFO] chosenVisLayers={chosenVisLayers if chosenVisLayers is not None else 'ALL (0-23)'}")
    print(f"[INFO] includeMerger={includeMerger}")
    print(f"[INFO] standardDivCutOff={standardDivCutOff}")
    print(f"[INFO] numRandomVarieties={numRandomVarieties}\n")

    pil = Image.open(IMAGE_PATH).convert("RGB")
    x_orig01 = pil_to_tensor01(pil).to(device)

    template_inputs = build_template_inputs(tokenizer, QUESTION, device)

    pv_clean = llava_preprocess_differentiable(x_orig01, image_processor)

    print("\n=== CLEAN OUTPUT ===")
    clean_text = run_generation_with_pixel_values(
        model,
        tokenizer,
        template_inputs,
        pv_clean,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(clean_text)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    conv_path = (
        f"llava_attack/outputsStorageImagenet/convergence/{attackSample}/"
        f"llava_ORIG_attack_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
        f"num_steps_{num_steps}_standardDivCutOff_{standardDivCutOff}_numRandomVarieties_{numRandomVarieties}_"
        f"ChosenSingVecs_lan-{lanLayersTag}_vis-{visLayersTag}_merger-{includeMerger}.npy"
    )

    x_adv01, best_pert = adam_attack_original_space(
        model=model,
        image_processor=image_processor,
        template_inputs=template_inputs,
        x_orig01=x_orig01,
        attck_type=attck_type,
        num_steps=num_steps,
        lr=lr,
        epsilon=epsilon,
        device=device,
        save_conv_path=conv_path,
        AttackStartLayer=AttackStartLayer,
        numLayerstAtAtime=numLayerstAtAtime,
        standardDivCutOff=standardDivCutOff,
        numRandomVarieties=numRandomVarieties,
        chosenLanLayers=chosenLanLayers,
        chosenVisLayers=chosenVisLayers,
        includeMerger=includeMerger,
    )

    adv_img_path = (
        f"llava_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
        f"num_steps_{num_steps}_standardDivCutOff_{standardDivCutOff}_numRandomVarieties_{numRandomVarieties}_"
        f"ChosenSingVecs_lan-{lanLayersTag}_vis-{visLayersTag}_merger-{includeMerger}.png"
    )

    adv_noise_path = (
        f"llava_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
        f"num_steps_{num_steps}_standardDivCutOff_{standardDivCutOff}_numRandomVarieties_{numRandomVarieties}_"
        f"ChosenSingVecs_lan-{lanLayersTag}_vis-{visLayersTag}_merger-{includeMerger}.pt"
    )

    tensor01_to_pil(x_adv01).save(adv_img_path)
    print(f"\nSaved ORIGINAL-resolution adversarial image to: {adv_img_path}")

    torch.save(best_pert.detach().cpu(), adv_noise_path)

    pv_adv = llava_preprocess_differentiable(x_adv01, image_processor)
    print("\n=== ADVERSARIAL OUTPUT ===")
    adv_text = run_generation_with_pixel_values(
        model,
        tokenizer,
        template_inputs,
        pv_adv,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(adv_text)

    cleanOutTxt = f"llava_attack/outputsStorageImagenet/advOutputs/{attackSample}/cleanOutput.txt"
    with open(cleanOutTxt, "w") as f:
        f.write(clean_text + "\n\n")

    advOutTxt = (
        f"llava_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
        f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
        f"num_steps_{num_steps}_standardDivCutOff_{standardDivCutOff}_numRandomVarieties_{numRandomVarieties}_"
        f"ChosenSingVecs_lan-{lanLayersTag}_vis-{visLayersTag}_merger-{includeMerger}.txt"
    )
    with open(advOutTxt, "w") as f:
        f.write(adv_text + "\n")

    print(f"\nSaved outputs to: {advOutTxt}")
    print(f"Saved convergence to: {conv_path}")


if __name__ == "__main__":
    main()
