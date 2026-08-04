


'''

Exactly the same attack as QwenUntargted_ChosenSingulaeVectors.py, EXCEPT
the attention query/key/value projections (vision fused qkv AND language
self_attn.q_proj/k_proj/v_proj) are entirely removed from the picture:

  - They are not collected as targets.
  - No forward hooks are registered on them.
  - They take no part in the singular-vector importance sampling
    (the clean-vs-random-noise weak_norm / standardDivCutOff selection).
  - They contribute nothing to the adversarial alignment loss.

Everything else -- attention OUTPUT proj (attn.proj / self_attn.o_proj),
every MLP projection (gate/up/down, vision AND language), and the
vision-to-language merger projection -- is targeted exactly as in
QwenUntargted_ChosenSingulaeVectors.py:

  1. For every remaining targeted linear operator, compute the FULL
     right-singular-vector basis of its weight
     (torch.linalg.svd(..., full_matrices=True)), matching
     getTopRightSingularVector from
     qwen/Qwen2p5SpectrumGuidedAttackSameSample.py.

  2. Run ONE clean forward pass (delta = 0) and `--numRandomVarieties`
     (default 10) forward passes with small random L_inf-epsilon noise
     deltas. Project each targeted operator's captured input activations
     onto every vector of its full singular basis
     (getMeanAlignmentWithTopRightSingularVector-style alignment energy),
     and measure, per singular-vector index, how much the alignment
     coefficient shifts between the clean pass and each noisy pass:

         diffWeak = sqrt(mean_over_tokens((clean_coeffs - weak_coeffs)^2))

     averaged over the `numRandomVarieties` noisy passes -> `weak_mean`,
     then normalized: `weak_norm = weak_mean / weak_mean.mean()`.

  3. Exactly the formula from Qwen2p5SpectrumGuidedAttackSameSample.py:

         indices = torch.where(weak_norm > standardDivCutOff * weak_norm.std())[0]

     selects the singular-vector indices whose alignment is unusually
     sensitive to small perturbations. Those selected vectors span the
     subspace the attack aligns the input activations toward.

  4. The optimization loop is unchanged: Adam over an original-image-space
     perturbation delta, minimizing the mean alignment-energy loss between
     each remaining targeted operator's input activations and its chosen
     singular-vector subspace, summed over every language / vision /
     merger target (with query/key/value simply absent from all three
     buckets).

--standardDivCutOff is the attack's spectral selection hyperparameter and
is recorded in every output filename.

Since attention q/k/v is removed, there is no per-attention-head splitting
anywhere in this file (that machinery only existed to slice the fused
vision qkv weight and the language q/k/v weights into their per-head
sub-matrices) -- every remaining target is a single whole weight matrix.

Example runs:



export CUDA_VISIBLE_DEVICES=0
conda deactivate
cd spectralShift/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
for ATTACK_SAMPLE in $(seq 1 50); do
    python qwen/QwenUntargted_ChosenNonAttnSingularVectors.py --attck_type noAttn --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --standardDivCutOff 3
done



export CUDA_VISIBLE_DEVICES=1
conda deactivate
cd spectralShift/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
for ATTACK_SAMPLE in $(seq 1 50); do
    python qwen/QwenUntargted_ChosenNonAttnSingularVectors.py --attck_type noAttn --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --standardDivCutOff 4
done



export CUDA_VISIBLE_DEVICES=2
conda deactivate
cd spectralShift/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
for ATTACK_SAMPLE in $(seq 1 50); do
    python qwen/QwenUntargted_ChosenNonAttnSingularVectors.py --attck_type noAttn --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --standardDivCutOff 5
done


export CUDA_VISIBLE_DEVICES=3
conda deactivate
cd spectralShift/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
for ATTACK_SAMPLE in $(seq 1 50); do
    python qwen/QwenUntargted_ChosenNonAttnSingularVectors.py --attck_type noAttn --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --standardDivCutOff 6
done





# Restrict to a subset of layers / fewer random noise varieties if desired:
python qwen/QwenUntargted_ChosenNonAttnSingularVectors.py --attck_type saa_loop --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 --attackSample 1 --standardDivCutOff 3 --numRandomVarieties 10 --chosenLanLayers 2 --chosenVisLayers 0 1 2 4 5 6 7 8 9 14 24

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


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
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


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
# Qwen differentiable preprocessing
# ----------------------------
def _get_qwen_resize_hw(image_processor, H: int, W: int):
    patch_size = int(getattr(image_processor, "patch_size", 14))
    merge_size = int(getattr(image_processor, "merge_size", 2))
    factor = patch_size * merge_size

    min_pixels = int(getattr(image_processor, "min_pixels", 56 * 56))
    max_pixels = int(getattr(image_processor, "max_pixels", 28 * 28 * 1280))

    def round_by_factor(x, f):
        return int(round(x / f) * f)

    def floor_by_factor(x, f):
        return int(np.floor(x / f) * f)

    def ceil_by_factor(x, f):
        return int(np.ceil(x / f) * f)

    h_bar = max(factor, round_by_factor(H, factor))
    w_bar = max(factor, round_by_factor(W, factor))

    if h_bar * w_bar > max_pixels:
        beta = np.sqrt((H * W) / max_pixels)
        h_bar = max(factor, floor_by_factor(H / beta, factor))
        w_bar = max(factor, floor_by_factor(W / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = np.sqrt(min_pixels / (H * W))
        h_bar = max(factor, ceil_by_factor(H * beta, factor))
        w_bar = max(factor, ceil_by_factor(W * beta, factor))

    return int(h_bar), int(w_bar)


def qwen_preprocess_differentiable(x01: torch.Tensor, processor):
    """
    Differentiable Qwen image preprocessing.
    Mirrors Qwen2.5-VL image patchification so gradients flow from model
    pixel_values back to original image space.
    """
    ip = processor.image_processor
    _, C, H, W = x01.shape
    assert C == 3, "Expected RGB tensor with 3 channels."

    patch_size = int(ip.patch_size)
    temporal_patch_size = int(ip.temporal_patch_size)
    merge_size = int(ip.merge_size)

    target_h, target_w = _get_qwen_resize_hw(ip, H, W)

    x = F.interpolate(x01, size=(target_h, target_w), mode="bilinear", align_corners=False)

    mean = torch.tensor(ip.image_mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(ip.image_std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std

    x = x.repeat(temporal_patch_size, 1, 1, 1)  # (T,C,H,W)

    grid_t = x.shape[0] // temporal_patch_size
    grid_h = target_h // patch_size
    grid_w = target_w // patch_size

    patches = x.view(
        grid_t,
        temporal_patch_size,
        3,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )

    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()
    pixel_values = patches.view(
        grid_t * grid_h * grid_w,
        3 * temporal_patch_size * patch_size * patch_size,
    )

    image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long, device=x01.device)
    return pixel_values, image_grid_thw


# ----------------------------
# Build template inputs once
# ----------------------------
def build_template_inputs(processor, question: str, pil_image: Image.Image, device):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    template = processor(
        text=[prompt],
        images=[pil_image],
        return_tensors="pt",
    )
    template = {k: v.to(device) if torch.is_tensor(v) else v for k, v in template.items()}
    return template


# ----------------------------
# Generation helper
# ----------------------------
def run_generation_with_pixel_values(model, processor, template_inputs, pixel_values, image_grid_thw, max_new_tokens=128):
    model.eval()
    inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
    inputs["pixel_values"] = pixel_values
    inputs["image_grid_thw"] = image_grid_thw

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_ids = inputs["input_ids"]
    gen_only = out_ids[:, input_ids.shape[1]:]
    return processor.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]


# ============================================================
# Chosen-singular-vector selected module attack for Qwen2.5-VL
# (attention query/key/value EXCLUDED everywhere)
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


MERGER_NAME_PATTERN = re.compile(r"visual\.merger\.mlp\.(0|2)$")
MERGER_LABEL = "Vis-to-lan proj"


def is_merger_target(name: str) -> bool:
    return bool(MERGER_NAME_PATTERN.search(name))


# ----------------------------
# Target-module discovery (every language layer, every vision block, merger)
# -- attention query/key/value projections are intentionally NOT collected.
# ----------------------------
def collect_target_specs(
    model,
    chosen_lan_layers=None,
    chosen_vis_layers=None,
    include_merger=True,
):
    """
    Same coverage as QwenUntargted_ChosenSingulaeVectors.py's
    collect_target_specs, EXCEPT the vision fused qkv projection and the
    language self_attn.q_proj/k_proj/v_proj projections are skipped
    entirely. Only attention OUTPUT proj (attn.proj / self_attn.o_proj),
    every MLP projection, and the merger remain -- all as whole weight
    matrices (no per-attention-head splitting is needed since there is no
    attention q/k/v left to split).
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
        if lan_idx is not None and "visual" not in name:
            if chosen_lan_layers_set is not None and lan_idx not in chosen_lan_layers_set:
                continue

            # NOTE: self_attn.q_proj / k_proj / v_proj are intentionally
            # NOT matched here -- query/key/value take no part in this
            # attack variant.
            if name.endswith(".self_attn.o_proj"):
                specs.append(dict(name=name, hook_name=name, module=module, kind="language",
                                   sub_kind="att output proj (lan)", layer_idx=lan_idx, weight_slice=None))
            elif name.endswith(".mlp.gate_proj"):
                specs.append(dict(name=name, hook_name=name, module=module, kind="language",
                                   sub_kind="MLP gate proj", layer_idx=lan_idx, weight_slice=None))
            elif name.endswith(".mlp.up_proj"):
                specs.append(dict(name=name, hook_name=name, module=module, kind="language",
                                   sub_kind="MLP up proj", layer_idx=lan_idx, weight_slice=None))
            elif name.endswith(".mlp.down_proj"):
                specs.append(dict(name=name, hook_name=name, module=module, kind="language",
                                   sub_kind="MLP down proj", layer_idx=lan_idx, weight_slice=None))
            continue

        # ---------------- vision transformer blocks ----------------
        vis_idx = extract_vision_layer_idx(name)
        if vis_idx is not None and "visual" in name:
            if chosen_vis_layers_set is not None and vis_idx not in chosen_vis_layers_set:
                continue

            # NOTE: attn.qkv (fused query/key/value) is intentionally NOT
            # matched here -- query/key/value take no part in this attack
            # variant.
            if name.endswith(".attn.proj"):
                specs.append(dict(name=name, hook_name=name, module=module, kind="vision",
                                   sub_kind="att output proj (vis)", layer_idx=vis_idx, weight_slice=None))
            elif name.endswith(".mlp.gate_proj"):
                specs.append(dict(name=name, hook_name=name, module=module, kind="vision",
                                   sub_kind="MLP gate (vis)", layer_idx=vis_idx, weight_slice=None))
            elif name.endswith(".mlp.up_proj"):
                specs.append(dict(name=name, hook_name=name, module=module, kind="vision",
                                   sub_kind="MLP up (vis)", layer_idx=vis_idx, weight_slice=None))
            elif name.endswith(".mlp.down_proj"):
                specs.append(dict(name=name, hook_name=name, module=module, kind="vision",
                                   sub_kind="MLP down (vis)", layer_idx=vis_idx, weight_slice=None))
            continue

        # ---------------- vision-to-language bridge ----------------
        if include_merger and is_merger_target(name):
            specs.append(dict(name=name, hook_name=name, module=module, kind="merger",
                               sub_kind=MERGER_LABEL, layer_idx=None, weight_slice=None))

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
# Full singular-vector basis (matches getTopRightSingularVector from
# Qwen2p5SpectrumGuidedAttackSameSample.py; no per-head variant needed
# here since there is no attention q/k/v left to split).
# ----------------------------
def compute_full_vh_for_spec(spec):
    with torch.no_grad():
        W = spec["module"].weight.detach().to(torch.float32)
        if spec["weight_slice"] is not None:
            start, end = spec["weight_slice"]
            W = W[start:end, :]
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


def _coeffs_against_full_vh(H, Vh):
    """
    Mirrors getMeanAlignmentWithTopRightSingularVector, returning the raw
    per-token, per-singular-vector coefficients.
    """
    H = _flatten_tokens(H)
    H_hat = F.normalize(H, dim=1)
    V_hat = F.normalize(Vh, dim=1)  # (n, n)
    coeffs = H_hat @ V_hat.T        # (N, n)
    return coeffs


def getMeanAlignmentLossWithChosenSubspace(InputToLayer, chosenRightSingularVectors):
    """
    Same alignment-energy loss as
    QwenUntargted_ChosenSingulaeVectors.py's
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
# -- run only over the non-attention-q/k/v targets collected above.
# ----------------------------
def run_importance_sampling(
    model,
    processor,
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

        pv, grid = qwen_preprocess_differentiable(x_p01, processor)
        inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
        inputs["pixel_values"] = pv
        inputs["image_grid_thw"] = grid
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

    print(f"\n[INFO] Importance sampling (query/key/value EXCLUDED): 1 clean pass + "
          f"{numRandomVarieties} weak-noise passes (epsilon={epsilon}) over "
          f"{len(unique_hook_names)} hooked modules ...")

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
    print("\n========== CHOSEN SINGULAR-VECTOR SUBSPACES (NO ATTENTION Q/K/V, ALL BLOCKS) ==========")

    num_kept = 0
    for spec_i, spec in enumerate(target_specs):
        hn = spec["hook_name"]
        if hn not in clean_snapshot:
            spec["target_vectors"] = None
            continue

        Vh = compute_full_vh_for_spec(spec).to(device)
        n = Vh.shape[-1]

        H_clean = clean_snapshot[hn].to(device)
        orig_coeffs = _coeffs_against_full_vh(H_clean, Vh)

        diffs = []
        for v in range(numRandomVarieties):
            H_weak = weak_snapshots[v][hn].to(device)
            weak_coeffs = _coeffs_against_full_vh(H_weak, Vh)
            diff_v = (orig_coeffs - weak_coeffs).pow(2).mean(dim=0).sqrt().reshape(-1)
            diffs.append(diff_v)
            del H_weak, weak_coeffs

        diffs = torch.stack(diffs, dim=0)             # (numRandomVarieties, n)
        weak_mean = diffs.mean(dim=0)                 # (n,)
        weak_mean_avg = weak_mean.mean()
        weak_norm = weak_mean / (weak_mean_avg + 1e-12)
        cutoff = standardDivCutOff * weak_norm.std()

        chosen_flat_idx = torch.where(weak_norm > cutoff)[0]
        if chosen_flat_idx.numel() == 0:
            # Safety fallback: always keep at least one (the most sensitive) direction.
            chosen_flat_idx = torch.argmax(weak_norm).view(1)

        target_vectors = Vh[chosen_flat_idx, :].contiguous()

        spec["target_vectors"] = target_vectors.detach().clone()
        spec["num_chosen"] = int(chosen_flat_idx.numel())
        num_kept += 1

        layer_str = f"{spec['layer_idx']:3d}" if spec["layer_idx"] is not None else "N/A"
        print(
            f"{spec['kind']:8s} | layer={layer_str} | {spec['sub_kind']:24s} | {hn} | "
            f"full_basis={n} | weak_norm.std={float(weak_norm.std().item()):.6f} "
            f"| chosen={spec['num_chosen']}"
        )

        del Vh, H_clean, orig_coeffs, diffs
        if device.type == "cuda" and spec_i % 8 == 0:
            torch.cuda.empty_cache()

    print("=========================================================================================")
    print(f"Total targets with a chosen subspace: {num_kept} / {len(target_specs)}")
    print("=========================================================================================\n")

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
    processor,
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

    target_specs = collect_target_specs(
        model,
        chosen_lan_layers=chosenLanLayers,
        chosen_vis_layers=chosenVisLayers,
        include_merger=includeMerger,
    )

    if len(target_specs) == 0:
        raise RuntimeError(
            "No target modules found. Check --chosenLanLayers and --chosenVisLayers."
        )

    hook_handles = register_all_hooks(target_specs)

    target_specs = run_importance_sampling(
        model,
        processor,
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

        pv_adv, grid_adv = qwen_preprocess_differentiable(x_adv01, processor)
        adv_inputs["pixel_values"] = pv_adv
        adv_inputs["image_grid_thw"] = grid_adv

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

        del outputs, loss, attack_loss, pv_adv, grid_adv
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
        description="Qwen2.5-VL ORIGINAL-image-space adversarial attack, importance-sampled singular-vector subspaces, ALL blocks, attention q/k/v EXCLUDED"
    )
    parser.add_argument("--attck_type", type=str, default="grill_l2", help="kept for compatibility")
    parser.add_argument("--desired_norm_l_inf", type=float, default=0.03, help="epsilon L_inf in ORIGINAL pixel space [0..1]")
    parser.add_argument("--learningRate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--num_steps", type=int, default=None, help="Number of Adam steps")
    parser.add_argument("--numSteps", type=int, default=None, help="Number of Adam steps, old Qwen-style name")
    parser.add_argument("--attackSample", type=str, default="nature", help="which sample")
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
        help="Number of random weak-noise passes used to estimate weak_norm (matches NumRandomVarieties in the reference script).",
    )

    # Backward-compatible single-layer arg from old Qwen script.
    parser.add_argument("--AlignLayer", type=int, default=None, help="old Qwen arg; used as chosenLanLayers if --chosenLanLayers is omitted")

    parser.add_argument(
        "--chosenLanLayers",
        type=int,
        nargs="+",
        default=None,
        help="Space-separated language layer indices to attack. Omit for ALL language layers (0-27).",
    )
    parser.add_argument(
        "--chosenVisLayers",
        type=int,
        nargs="+",
        default=None,
        help="Space-separated vision layer indices to attack. Omit for ALL vision blocks (0-31).",
    )
    parser.add_argument(
        "--includeMerger",
        dest="includeMerger",
        action="store_true",
        default=True,
        help="Include the vision-to-language merger projection (default: on).",
    )
    parser.add_argument(
        "--no-includeMerger",
        dest="includeMerger",
        action="store_false",
        help="Exclude the vision-to-language merger projection.",
    )

    # NOTE: attention query/key/value are always excluded in this script --
    # there is intentionally no flag to re-enable them here. Use
    # QwenUntargted_ChosenSingulaeVectors.py for the variant that includes them.

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

    # Preserve Qwen paths.
    MODEL_PATH = "../illcond/QwenAttack/Qwen2.5-VL-7B-Instruct"
    IMAGE_PATH = f"../interpretAttacks/llava_attack/dataSamplesForQuant/{attackSample}.JPEG"
    QUESTION = "What is shown in this image?"
    MAX_NEW_TOKENS = 128

    os.makedirs("qwen/outputsStorageImagenet", exist_ok=True)
    os.makedirs(f"qwen/outputsStorageImagenet/advOutputs/{attackSample}", exist_ok=True)
    os.makedirs(f"qwen/outputsStorageImagenet/convergence/{attackSample}", exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"device={device}, dtype={dtype}")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    print("\n[INFO] Qwen ALL-block, CHOSEN-singular-vector variant, ATTENTION Q/K/V EXCLUDED.")
    print("[INFO] Only attn output proj, every MLP projection (vision + language), and the")
    print("[INFO] merger projection are targeted. Singular-vector indices are importance-")
    print("[INFO] sampled via the clean-vs-random-noise sensitivity criterion from")
    print("[INFO] Qwen2p5SpectrumGuidedAttackSameSample.py, then used to span the alignment")
    print("[INFO] target subspace for each remaining operator.")
    print(f"[INFO] chosenLanLayers={chosenLanLayers if chosenLanLayers is not None else 'ALL (0-27)'}")
    print(f"[INFO] chosenVisLayers={chosenVisLayers if chosenVisLayers is not None else 'ALL (0-31)'}")
    print(f"[INFO] includeMerger={includeMerger}")
    print(f"[INFO] standardDivCutOff={standardDivCutOff}")
    print(f"[INFO] numRandomVarieties={numRandomVarieties}\n")

    pil = Image.open(IMAGE_PATH).convert("RGB")
    x_orig01 = pil_to_tensor01(pil).to(device)

    template_inputs = build_template_inputs(processor, QUESTION, pil, device)

    pv_clean, grid_clean = qwen_preprocess_differentiable(x_orig01, processor)

    print("\n=== CLEAN OUTPUT ===")
    clean_text = run_generation_with_pixel_values(
        model,
        processor,
        template_inputs,
        pv_clean,
        grid_clean,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(clean_text)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    conv_path = (
        f"qwen/outputsStorageImagenet/convergence/{attackSample}/"
        f"qwen_ORIG_attack_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
        f"num_steps_{num_steps}_standardDivCutOff_{standardDivCutOff}_numRandomVarieties_{numRandomVarieties}_"
        f"ChosenNonAttnSingVecs_lan-{lanLayersTag}_vis-{visLayersTag}_merger-{includeMerger}.npy"
    )

    x_adv01, best_pert = adam_attack_original_space(
        model=model,
        processor=processor,
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
        f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
        f"num_steps_{num_steps}_standardDivCutOff_{standardDivCutOff}_numRandomVarieties_{numRandomVarieties}_"
        f"ChosenNonAttnSingVecs_lan-{lanLayersTag}_vis-{visLayersTag}_merger-{includeMerger}.png"
    )

    adv_noise_path = (
        f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
        f"num_steps_{num_steps}_standardDivCutOff_{standardDivCutOff}_numRandomVarieties_{numRandomVarieties}_"
        f"ChosenNonAttnSingVecs_lan-{lanLayersTag}_vis-{visLayersTag}_merger-{includeMerger}.pt"
    )

    tensor01_to_pil(x_adv01).save(adv_img_path)
    print(f"\nSaved ORIGINAL-resolution adversarial image to: {adv_img_path}")

    torch.save(best_pert.detach().cpu(), adv_noise_path)

    pv_adv, grid_adv = qwen_preprocess_differentiable(x_adv01, processor)
    print("\n=== ADVERSARIAL OUTPUT ===")
    adv_text = run_generation_with_pixel_values(
        model,
        processor,
        template_inputs,
        pv_adv,
        grid_adv,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(adv_text)

    cleanOutTxt = f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/cleanOutput.txt"
    with open(cleanOutTxt, "w") as f:
        f.write(clean_text + "\n\n")

    advOutTxt = (
        f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
        f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
        f"num_steps_{num_steps}_standardDivCutOff_{standardDivCutOff}_numRandomVarieties_{numRandomVarieties}_"
        f"ChosenNonAttnSingVecs_lan-{lanLayersTag}_vis-{visLayersTag}_merger-{includeMerger}.txt"
    )
    with open(advOutTxt, "w") as f:
        f.write(adv_text + "\n")

    print(f"\nSaved outputs to: {advOutTxt}")
    print(f"Saved convergence to: {conv_path}")


if __name__ == "__main__":
    main()
