'''

export CUDA_VISIBLE_DEVICES=3
conda deactivate
cd spectralShift/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
for StudyLayer in $(seq 0 27); do
    python qwen/Qwen2p5SpectrumGuidedAttackSameSampleEfficient.py --attck_type bsa --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --VisionLayerTrack 0 --LanLayerTrack $StudyLayer --kthSingVec -10 --attackMode lan
done





export CUDA_VISIBLE_DEVICES=1
conda deactivate
cd spectralShift/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
for StudyLayer in $(seq 0 31); do
    python qwen/Qwen2p5SpectrumGuidedAttackSameSampleEfficient.py --attck_type bsa --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --VisionLayerTrack $StudyLayer --LanLayerTrack 0 --kthSingVec 10 --attackMode vis
done



export CUDA_VISIBLE_DEVICES=1
conda deactivate
cd spectralShift/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
for StudyLayer in $(seq 0 31); do
    python qwen/Qwen2p5SpectrumGuidedAttackSameSampleEfficient.py --attck_type bsa --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --VisionLayerTrack $StudyLayer --LanLayerTrack 0 --kthSingVec 10 --attackMode mulmod
done





export CUDA_VISIBLE_DEVICES=1
conda deactivate
cd spectralShift/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
python qwen/Qwen2p5SpectrumGuidedAttackSameSampleEfficient.py --attck_type bsa --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --VisionLayerTrack 0 --LanLayerTrack 0 --kthSingVec 10 --attackMode vis

'''


# ==============================================================================================================================================================
# PERFORMANCE REWRITE NOTES (read me)
# ==============================================================================================================================================================
# This file produces the exact same 60 PNGs (32 "vis" + 1 "mulmod" + 27 "lan"),
# with the same filenames, same axes, same statistics (mean +/- std of the L2
# distance between "original" and "weak-adversary" spectral-alignment
# coefficients). Only *how* those numbers get computed was changed. The
# original script was extremely wasteful in a few concrete, fixable ways:
#
#   1. NO-OP GRADIENT TRACKING. `delta.requires_grad_(True)` was set on every
#      call, so every single forward pass built a full autograd graph through
#      the whole 7B-parameter model -- even though `.backward()` is never
#      called anywhere in this file (the "adam_attack" function is actually
#      just a single forward pass against a fixed perturbation). Building and
#      holding that graph roughly doubles compute/memory for nothing.
#      FIX: everything now runs under `torch.inference_mode()`.
#
#   2. A WASTED "CLEAN" FORWARD PASS. `adam_attack_original_space` ran a full
#      forward pass on the clean image (`outputsN`) purely to read
#      `len(outputsN.hidden_states)` for a bounds check that never fires in
#      practice. That's a whole extra 7B-model forward per call, i.e. HALF of
#      all compute in the original script was thrown away.
#      FIX: that length is now derived from `config.depth` /
#      `config.num_hidden_layers` directly -- no forward pass needed.
#
#   3. UNUSED LOSS + UNUSED HIDDEN STATES. `labels=input_ids` was passed
#      (forces a cross-entropy over the full ~150k vocab) and
#      `output_hidden_states=True` was requested, but neither the loss nor
#      the hidden-state tuple is ever read anywhere in this file -- only the
#      *hook-captured* intermediate activations matter.
#      FIX: labels removed, output_hidden_states left off, and (see #4) the
#      LM head itself is skipped for the "lan" hook path.
#
#   4. THE FULL MODEL WAS ALWAYS RUN, EVEN WHEN ONLY THE VISION TOWER OR ONLY
#      PART OF THE DECODER WAS NEEDED. `attackMode == "vis"` and
#      `"mulmod"` only ever hook into `model.model.visual` (confirmed against
#      the actual Qwen2.5-VL source: the vision tower's `forward` takes
#      `(hidden_states, grid_thw)` directly and is fully self-contained), so
#      running the entire 7B-parameter language decoder afterwards for those
#      two modes was pure waste. Likewise `attackMode == "lan"` only reads
#      activations up to `LanLayerTrack`, so decoder layers *after* that
#      index don't need to run at all (verified: `Qwen2_5_VLTextModel.forward`
#      just does `for i, layer in enumerate(self.layers): ...` with the
#      causal mask/rotary embeddings precomputed once beforehand,
#      independent of how many layers are actually looped over -- so
#      temporarily slicing `self.layers` is safe and fully restorable).
#      FIX: "vis"/"mulmod" call `model.model.visual(...)` directly (skips the
#      entire language model + the LM head). "lan" calls the base
#      `model.model(...)` (skips the LM head's huge vocab projection) with
#      the decoder's layer list temporarily truncated to `LanLayerTrack + 1`
#      layers, restored immediately after.
#
#   5. TEN SEQUENTIAL FORWARD PASSES INSTEAD OF ONE BATCHED ONE. For every
#      tracked layer, the script ran 1 "original" + 9 "weak adversary"
#      forward passes back-to-back. Qwen2.5-VL's vision tower is
#      token-flat and batches multiple images by concatenating their patches
#      (with `image_grid_thw` carrying one row per image); the language
#      decoder batches the usual way. Verified from source that the vision
#      tower's window-attention index permutation never mixes tokens across
#      images (`window_index_id` is offset per image, so each image's tokens
#      stay in one contiguous block), so the 10 variants can be run as ONE
#      batch-of-10 forward pass and the resulting per-token coefficients can
#      be split back into 10 contiguous per-sample chunks after the fact,
#      with identical semantics to 10 separate calls.
#      FIX: `run_batched_alignment_analysis(...)` builds all 10 perturbed
#      images (using the exact same sequential `torch.randn_like` draws as
#      the original, so the random deltas are bit-identical to before), runs
#      ONE batched forward, and splits the hook-captured coefficients back
#      into per-sample tensors before doing the same original->weak L2
#      distance math as before.
#
#   6. LEAKED FORWARD HOOKS. `getAllallTopRightSingularVectorsVis` /
#      `getallTopRightSingularVectorsLan` / `getallTopRightSingularVectorsMultiMod`
#      registered new hooks every layer without ever removing the previous
#      layer's hooks. Since the *entire* model is always run end-to-end, all
#      previously-registered hooks kept firing on every subsequent forward
#      pass for the rest of the run (hooks accumulate O(L) over the sweep).
#      FIX: these functions now return their hook handles, and every handle
#      is `.remove()`-d as soon as that layer's analysis is done.
#
# None of the plotted quantities, file names, or axis semantics were changed.
# The only numerically-visible difference is that batched-vs-sequential
# matmuls are not bit-exact (ordinary floating point non-associativity), well
# within the noise already being averaged over 9 random draws.
# ==============================================================================================================================================================


import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import sys
import csv
import argparse
import random
import contextlib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import matplotlib.pyplot as plt

# ----------------------------
# Reproducibility
# MODIFIED vs Gemma: ported verbatim from the existing, known-working Qwen
# scripts (QwenUntargeted_BSA.py / QwenUntargeted_BSA_inference.py) rather
# than reusing Gemma's simpler set_seed/backend settings, so this file
# behaves identically to the scripts that already run cleanly on this model.
# In particular, disabling flash/mem-efficient SDPA and forcing the math
# (eager) attention backend avoids the non-deterministic / fused-kernel
# behavior those backends can introduce.
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


criterion = nn.MSELoss()


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
    wasserstein_dist = torch.mean(torch.abs(tensor_a_sorted - tensor_b_sorted))
    return wasserstein_dist

# ----------------------------
# Losses: GRILL + OA
# (kept for parity with the original file; not used by the analysis path
# below -- neither `outputs.logits`/`hidden_states` from a "clean" forward
# nor these loss functions were ever actually called in the original script.)
# ----------------------------
def get_grill_l2(outputs, outputsN):
    loss = 0.0
    for h, hn in zip(outputs.hidden_states, outputsN.hidden_states):
        loss = loss + criterion(h, hn)
    return loss * criterion(h, hn)


def get_grill_wass(outputs, outputsN, startPos, endPos):
    loss = 0.0
    for h, hn in zip(outputs.hidden_states[startPos:endPos], outputsN.hidden_states[startPos:endPos]):
        loss = loss + wasserstein_distance(h, hn)
    return loss #* wasserstein_distance(h, hn)


def get_grill_cos(outputs, outputsN):
    loss = 0.0
    for h, hn in zip(outputs.hidden_states, outputsN.hidden_states):
        loss = loss + (1.0 - cos(h, hn)) ** 2
    return loss * (1.0 - cos(outputs.logits, outputsN.logits)) ** 2


def get_oa_l2(outputs, outputsN):
    return criterion(outputs.logits, outputsN.logits)


def get_oa_wass(outputs, outputsN):
    return wasserstein_distance(outputs.logits, outputsN.logits)


def get_oa_cos(outputs, outputsN):
    return (1.0 - cos(outputs.logits, outputsN.logits)) ** 2


# ----------------------------
# Utilities: image <-> tensor
# ----------------------------
def pil_to_tensor01(pil_img: Image.Image) -> torch.Tensor:
    """PIL RGB -> torch float tensor in [0,1], shape (1,3,H,W)"""
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    return t


def tensor01_to_pil(t01: torch.Tensor) -> Image.Image:
    """torch tensor [0,1], shape (1,3,H,W) or (3,H,W) -> PIL RGB"""
    if t01.dim() == 4:
        t01 = t01[0]
    t01 = t01.detach().cpu().clamp(0, 1)
    arr = (t01.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ----------------------------
# Differentiable preprocessing (approx Qwen2.5-VL)
# MODIFIED vs Gemma: Qwen's image processor does a dynamic-resolution resize
# (rounded to patch*merge multiples) and then repacks pixels into flattened
# patch tokens (pixel_values) plus an image_grid_thw tensor. This replaces
# Gemma's fixed-size resize+center-crop pipeline entirely, since Qwen2.5-VL
# has no such thing (necessary structural change, taken directly from
# qwen/QwenUntargeted_BSA_inference.py so the same perturbations line up).
# ----------------------------
def _get_qwen_resize_hw(image_processor, H, W):
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
    Single-image version (kept for reference / compatibility). The hot path
    below uses `qwen_preprocess_differentiable_batch` instead so that all 10
    (1 original + 9 weak) perturbations of a layer's analysis go through the
    vision tower / language model in a single batched forward pass instead
    of 10 sequential ones.
    """
    ip = processor.image_processor
    _, C, H, W = x01.shape
    assert C == 3

    patch_size = int(ip.patch_size)
    temporal_patch_size = int(ip.temporal_patch_size)
    merge_size = int(ip.merge_size)

    target_h, target_w = _get_qwen_resize_hw(ip, H, W)

    x = F.interpolate(x01, size=(target_h, target_w), mode="bilinear", align_corners=False)

    mean = torch.tensor(ip.image_mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(ip.image_std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std

    x = x.repeat(temporal_patch_size, 1, 1, 1)

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


def qwen_preprocess_differentiable_batch(x_batch01: torch.Tensor, processor):
    """
    OPTIMIZATION (see note #5 at top of file): batched version of
    `qwen_preprocess_differentiable`. Takes B images that all share the same
    (H, W) -- which is always true here, since the B images are just the
    original image plus 9 perturbed copies of it -- and returns a single
    `pixel_values` tensor with all B images' patches concatenated (exactly
    how Qwen2.5-VL expects multi-image batches to be packed) plus an
    `image_grid_thw` with one [grid_t, grid_h, grid_w] row per image.

    This produces bit-identical per-image patch tokens to calling
    `qwen_preprocess_differentiable` B times and concatenating the results
    (same resize target for every image since H, W match; the only
    difference from the single-image function is that the leading batch
    axis is threaded through the reshape/permute instead of looped).
    """
    ip = processor.image_processor
    B, C, H, W = x_batch01.shape
    assert C == 3

    patch_size = int(ip.patch_size)
    temporal_patch_size = int(ip.temporal_patch_size)
    merge_size = int(ip.merge_size)

    target_h, target_w = _get_qwen_resize_hw(ip, H, W)

    x = F.interpolate(x_batch01, size=(target_h, target_w), mode="bilinear", align_corners=False)

    mean = torch.tensor(ip.image_mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(ip.image_std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std  # (B,3,th,tw)

    grid_t = 1
    grid_h = target_h // patch_size
    grid_w = target_w // patch_size

    # Duplicate each sample's single frame across the temporal-patch axis
    # (same trick the original single-image code used via `.repeat`), but
    # keep the B samples distinct instead of merging them into the batch dim.
    x = x.unsqueeze(1).repeat(1, temporal_patch_size, 1, 1, 1)  # (B,T,3,th,tw)

    patches = x.view(
        B,
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

    patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).contiguous()

    pixel_values = patches.view(
        B * grid_t * grid_h * grid_w,
        3 * temporal_patch_size * patch_size * patch_size,
    )

    image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]] * B, dtype=torch.long, device=x_batch01.device)

    return pixel_values, image_grid_thw


# ----------------------------
# Build template inputs ONCE (IMPORTANT)
# Ensures image placeholder tokens exist in input_ids
# ----------------------------
def build_template_inputs(processor, question: str, pil_image: Image.Image, device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    # Pass an image ONCE so the processor inserts the correct special image token(s)
    template = processor(text=[prompt], images=[pil_image], return_tensors="pt")
    template = {k: v.to(device) if torch.is_tensor(v) else v for k, v in template.items()}
    return template


# ----------------------------
# Generation helper (uses template, swaps pixel_values / image_grid_thw)
# MODIFIED vs Gemma: Qwen2.5-VL needs image_grid_thw alongside pixel_values.
# (Not used by the analysis path below -- kept for parity / potential reuse.)
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
            do_sample=False,   # deterministic
        )

    input_ids = inputs["input_ids"]
    gen_only = out_ids[:, input_ids.shape[1]:]
    return processor.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]


def _drop_batch_dim_if_present(t):
    """
    ADDED vs Gemma (necessary correctness fix): Gemma's SigLIP vision tower
    and its language model both keep a leading batch dimension of size 1, so
    the original code's `InputToLayer[0]` reliably drops that batch dim and
    leaves a (N, d_model) tensor. Qwen2.5-VL's vision transformer instead
    runs on a single FLAT (total_tokens, hidden_dim) tensor with NO batch
    dimension at all (patches from the image are concatenated along dim 0
    directly).

    GENERALIZED for batched analysis (note #5): rather than only squeezing
    an exact batch-size-of-1 leading dim, this now reshapes ANY (B, S, d)
    tensor down to (B*S, d) via `.reshape(-1, d)`. For B == 1 this is
    numerically identical to the old `t[0]` behavior (same memory layout),
    and for B > 1 it flattens batch-major -- i.e. sample 0's S tokens first,
    then sample 1's, etc. -- which is exactly the layout the batched callers
    below rely on when splitting the resulting per-token coefficients back
    into per-sample chunks.
    """
    if t.dim() == 3:
        return t.reshape(-1, t.shape[-1])
    return t


def getMeanAlignmentWithTopRightSingularVector(InputToLayer, topRightSingularVector):


    H = _drop_batch_dim_if_present(InputToLayer)
    V = topRightSingularVector
    V = V.to(H)

    H_hat = F.normalize(H, dim=1)
    V_hat = F.normalize(V, dim=1)

    coeffs = H_hat @ V_hat.T        # (N, k)

    # Energy per token (sum of squared coefficients)
    per_token_energy = (coeffs ** 2).sum(dim=1)  # (N,)

    #print("coeffs.shape", coeffs.shape)
    #print("coeffs.mean(dim=1).shape", coeffs.mean(dim=1).shape)

    #coeffHIst = coeffs.mean(dim=1)

    # Mean across tokens
    mean_energy = per_token_energy.mean().item()

    # MODIFIED: return the raw (un-reduced) coeffs instead of coeffs.mean(dim=1).
    # We need the pre-reduction coeffs so that, at the call site, we can compute
    # an L2 difference against another (original/weak/strong) coeffs tensor
    # BEFORE doing the token-wise averaging.
    return mean_energy, coeffs

def getMeanAlignmentWithTopLeftSingularVector(InputToLayer, topRightSingularVector):
    v = topRightSingularVector.to(InputToLayer)
    v_hat = v / v.norm()
    InputToLayer = _drop_batch_dim_if_present(InputToLayer)
    h_hat = InputToLayer / InputToLayer.norm(dim=-1, keepdim=True)
    dots = h_hat @ v_hat
    dots = dots.squeeze(0)
    mean_abs_value = dots.abs().mean().item()
    return mean_abs_value


def getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, topRightSingularVector):
    #print("InputToLayer.shape", InputToLayer.shape)
    H = _drop_batch_dim_if_present(InputToLayer)          # (N, d_model)
    V = topRightSingularVector   # (num_heads, k, d_model)
    V = V.to(H)
    #print("H.shape", H.shape)
    #print("V.shape", V.shape)
    H_hat = F.normalize(H, dim=1)        # (N, d_model)
    V_hat = F.normalize(V, dim=2)        # (num_heads, k, d_model)


    #print("H_hat.shape", H_hat.shape)
    #print("V_hat.shape", V_hat.shape)

    coeffs = torch.einsum('nd,hkd->hnk', H_hat, V_hat) # this gave me the dot product between "all the bottom k singular vectors from all the heads on the weight matrix", and all the input tokens. Now I have so many dot product values (num_heads x N) number of dot prducts executed and values stored.

    #print("coeffs.shape", coeffs.shape)
    #now I will perform l2 norm of the dot products corresponding to all the nottom k singular vectors. That is nothing but squaring those singular vectors and adding. This equivalent to l2 norm , squaring,  adding, square rooting and then again squaring
    energy = (coeffs ** 2).sum(dim=2) # right now I have per-token, per-head subspace energy.
    mean_energy_per_head = energy.mean(dim=1)  # (num_heads,)
    mean_energy_all = mean_energy_per_head.mean().item()
    #dots = H_hat @ V_hat.T               # (N, num_heads)
    #mean_abs_value = dots.abs().mean().item()

    # MODIFIED: return the raw (un-reduced) coeffs (shape h,n,k) instead of
    # coeffs.mean(dim=1). Same reasoning as above: we need the pre-reduction
    # tensor to compute an L2 difference against another coeffs tensor first.
    # NOTE: this helper is fully generic in num_heads (it just reads V's own
    # shape), so it works unchanged for Qwen's vision heads (MHA, 16 heads)
    # AND for the language-model query heads (28) as well as the GQA
    # key/value heads (4) -- no modification needed here.
    return mean_energy_all, coeffs


def getMeanAlignmentWithAttentionHeadTopLeftSingularVector(OutputOfLayer, num_headsT, topLeftSingularVector):

    H = _drop_batch_dim_if_present(OutputOfLayer)          # (N, d_model)
    V = topLeftSingularVector   # (num_heads, d_model)
    V = V.to(H)
    H_hat = F.normalize(H, dim=1)        # (N, d_model)
    V_hat = F.normalize(V, dim=1)        # (num_heads, d_model)
    #print()

    H_hat = H_hat.view(H_hat.shape[0], num_headsT, -1)
    #print("after")
    #print("H_hat.shape", H_hat.shape)
    #print("V_hat.shape", V_hat.shape)
    dots = H_hat @ V_hat.T               # (N, num_heads)
    mean_abs_value = dots.abs().mean().item()

    return mean_abs_value


def getMeanAlignmentWithLanAttentionHeadTopRightSingularVector(InputToLayer, topRightSingularVector):
    #print("InputToLayer.shape", InputToLayer.shape)
    H = _drop_batch_dim_if_present(InputToLayer)          # (N, d_model)
    V = topRightSingularVector   # (num_heads, d_model)
    V = V.to(H)
    H_hat = F.normalize(H, dim=1)        # (N, d_model)
    V_hat = F.normalize(V, dim=1)        # (num_heads, d_model)
    #print("V_hat.shape", V_hat.shape)
    #print("H_hat.shape", H_hat.shape)
    dots = H_hat @ V_hat.T               # (N, num_heads)
    mean_abs_value = dots.abs().mean().item()

    return mean_abs_value


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# MODIFIED vs Gemma: robust getters for the vision blocks / language layers /
# vision-to-language merger. Different transformers versions expose Qwen2.5-VL
# with or without an extra ".model." wrapper, so we defensively probe for both
# (this mirrors the same hasattr() pattern already used in
# qwen/QwenUntargeted_BSA.py's run_get_image_features_with_vision_hooks()).
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
def get_vision_module_and_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "visual") and hasattr(model.model.visual, "blocks"):
        return model.model.visual, model.model.visual.blocks
    elif hasattr(model, "visual") and hasattr(model.visual, "blocks"):
        return model.visual, model.visual.blocks
    else:
        raise RuntimeError("Could not find Qwen2.5-VL vision blocks.")


def get_language_module_and_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model, model.model.language_model.layers
    elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model, model.language_model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model, model.model.layers
    else:
        raise RuntimeError("Could not find Qwen2.5-VL language-model layers.")


def get_base_multimodal_module(model):
    """
    OPTIMIZATION (note #4): returns the base `Qwen2_5_VLModel` (i.e.
    `model.model`) so "lan"-mode forward passes can skip the LM head's
    hidden_size x vocab_size projection entirely -- that projection's output
    (`logits`) is never read anywhere in this file, only the hook-captured
    intermediate activations are.
    """
    if hasattr(model, "model"):
        return model.model
    return model


def _resolve_vision_config(model):
    cfg = model.config
    return getattr(cfg, "vision_config", cfg)


def _resolve_text_config(model):
    cfg = model.config
    return getattr(cfg, "text_config", cfg)


@contextlib.contextmanager
def _truncated_module_list(owner, attr_name, upto):
    """
    OPTIMIZATION (note #4): temporarily replaces `owner.<attr_name>` (an
    `nn.ModuleList`) with just its first `upto` entries for the duration of
    the `with` block, then restores the original full list -- guaranteed via
    `finally`, even if the forward pass raises.

    Verified safe against the actual Qwen2.5-VL source for both the vision
    tower's `self.blocks` and the language model's `self.layers`: in both
    forward() implementations the loop is a plain
    `for i, item in enumerate(self.<list>): ...`, with every other
    layer-count-independent quantity (rotary embeddings, causal/window
    attention masks, cu_seqlens) computed once *before* that loop from
    `grid_thw` / `config` / sequence length alone -- never from
    `len(self.<list>)`. So truncating the list only skips the tail-end
    layers' compute; it does not change what the earlier (kept) layers
    compute, and nothing downstream of the loop is used by this script (the
    hook-captured intermediate activations are all that matter -- final
    outputs like `logits`/`pooler_output` from the truncated run are
    discarded).
    """
    full_list = getattr(owner, attr_name)
    truncated = nn.ModuleList(list(full_list)[:upto])
    setattr(owner, attr_name, truncated)
    try:
        yield truncated
    finally:
        setattr(owner, attr_name, full_list)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
qry0_inputs = {}
def qry0_pre_hook(module, inputs):
    qry0_inputs["qry0_in"] = inputs[0]
qry0_outputs = {}
def qry0_forward_hook(module, inputs, output):
    qry0_outputs["qry0_out"] = output
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

key0_inputs = {}
def key0_pre_hook(module, inputs):
    key0_inputs["key0_in"] = inputs[0]
key0_outputs = {}
def key0_forward_hook(module, inputs, output):
    key0_outputs["key0_out"] = output

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
val0_inputs = {}
def val0_pre_hook(module, inputs):
    val0_inputs["val0_in"] = inputs[0]
val0_outputs = {}
def val0_forward_hook(module, inputs, output):
    val0_outputs["val0_out"] = output
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

visOutProj_inputs = {}
def visOutProj_pre_hook(module, inputs):
    visOutProj_inputs["visOutProj_in"] = inputs[0]
visOutProj_outputs = {}
def visOutProj_forward_hook(module, inputs, output):
    visOutProj_outputs["visOutProj_out"] = output
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# MODIFIED vs Gemma: Gemma's vision MLP is a 2-layer FC (fc1/fc2). Qwen2.5-VL's
# vision MLP is a 3-projection SwiGLU block (gate_proj/up_proj/down_proj),
# exactly like the language-model MLP below. FC1/FC2 hooks are therefore
# replaced with three vision hooks: visGate / visUp / visDown.
visGate_inputs = {}
def visGate_pre_hook(module, inputs):
    visGate_inputs["visGate_in"] = inputs[0]
visGate_outputs = {}
def visGate_forward_hook(module, inputs, output):
    visGate_outputs["visGate_out"] = output
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
visUp_inputs = {}
def visUp_pre_hook(module, inputs):
    visUp_inputs["visUp_in"] = inputs[0]
visUp_outputs = {}
def visUp_forward_hook(module, inputs, output):
    visUp_outputs["visUp_out"] = output
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
visDown_inputs = {}
def visDown_pre_hook(module, inputs):
    visDown_inputs["visDown_in"] = inputs[0]
visDown_outputs = {}
def visDown_forward_hook(module, inputs, output):
    visDown_outputs["visDown_out"] = output
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

MulModProj_inputs = {}
def MulModProj_pre_hook(module, inputs):
    MulModProj_inputs["MulModProj_in"] = inputs[0]
MulModProj_outputs = {}
def MulModProj_forward_hook(module, inputs, output):
    MulModProj_outputs["MulModProj_out"] = output
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
qryLan_inputs = {}
def qryLan_pre_hook(module, inputs):
    qryLan_inputs["qryLan_in"] = inputs[0]
qryLan_outputs = {}
def qryLan_forward_hook(module, inputs, output):
    qryLan_outputs["qryLan_out"] = output
#----------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------
keyLan_inputs = {}
def keyLan_pre_hook(module, inputs):
    keyLan_inputs["keyLan_in"] = inputs[0]
keyLan_outputs = {}
def keyLan_forward_hook(module, inputs, output):
    keyLan_outputs["keyLan_out"] = output
#----------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------
valLan_inputs = {}
def valLan_pre_hook(module, inputs):
    valLan_inputs["valLan_in"] = inputs[0]
valLan_outputs = {}
def valLan_forward_hook(module, inputs, output):
    valLan_outputs["valLan_out"] = output
#----------------------------------------------------------------------------------------

GateLayerInputs = {}
def gate_proj_pre_hook(module, inputs):
    GateLayerInputs["gate_in"] = inputs[0]
GateLayerOutputs = {}
def gate_proj_forward_hook(module, inputs, output):
    GateLayerOutputs["gate_out"] = output
#----------------------------------------------------------------------------------------
UpLayerInputs = {}
def up_proj_pre_hook(module, inputs):
    UpLayerInputs["up_in"] = inputs[0]
UpLayerOutputs = {}
def up_proj_forward_hook(module, inputs, output):
    UpLayerOutputs["up_out"] = output
#----------------------------------------------------------------------------------------

DownLayer_inputs = {}
def down_proj_pre_hook(module, inputs):
    DownLayer_inputs["down_in"] = inputs[0]
DownLayer_outputs = {}
def down_proj_forward_hook(module, inputs, output):
    DownLayer_outputs["down_out"] = output

#----------------------------------------------------------------------------------------

def debug_hook(module, inputs, output):
    print("HOOKED:", module)
    if torch.is_tensor(output):
        print("output.shape:", output.shape)
    else:
        print("output type:", type(output))


def getTopRightSingularVector(down_proj):
    return torch.linalg.svd(down_proj.weight.to(torch.float32))[2][:]

def getTopRightSingularVectorForVisionQKV(qkvParam, which, d_model, d_head, num_heads):
    full_weight = qkvParam.weight.to(torch.float32)
    idx = {"q": 0, "k": 1, "v": 2}[which]
    weight_slice = full_weight[idx * d_model:(idx + 1) * d_model, :]
    weight_slice = weight_slice.view(num_heads, d_head, d_model)
    vh_per_head = []
    for h in range(num_heads):
        Wh = weight_slice[h]
        U, S, Vh = torch.linalg.svd(Wh.to(torch.float32))
        vh_per_head.append(Vh[:])
    return torch.stack(vh_per_head, 0)




def getTopRightSingularVectorForLanAttentioHeads(qryParam, num_heads_local, d_modelT):
    param = qryParam.weight.to(torch.float32)
    d_headT = param.shape[0] // num_heads_local
    param_heads = param.view(num_heads_local, d_headT, d_modelT)
    query_vh_per_head = []
    for h in range(num_heads_local):
        Wh = param_heads[h]
        U, S, Vh = torch.linalg.svd(Wh.to(torch.float32) )
        query_vh_per_head.append(Vh[:])
    query_vh_per_head = torch.stack(query_vh_per_head, 0)
    return query_vh_per_head

def getAllallTopRightSingularVectorsVis(vision_blocks, VisionLayerTrack, d_model, d_head, num_heads):
    """
    OPTIMIZATION (note #6): now returns the registered hook handles alongside
    the singular vectors, so the caller can `.remove()` them once this
    layer's analysis is finished instead of leaving them attached (and
    firing on every later layer's forward pass) for the rest of the run.
    """
    handles = []

    qkvParam = vision_blocks[VisionLayerTrack].attn.qkv
    handles.append(qkvParam.register_forward_pre_hook(qry0_pre_hook))
    handles.append(qkvParam.register_forward_hook(qry0_forward_hook))
    #------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------
    handles.append(qkvParam.register_forward_pre_hook(key0_pre_hook))
    handles.append(qkvParam.register_forward_hook(key0_forward_hook))
    #------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------
    handles.append(qkvParam.register_forward_pre_hook(val0_pre_hook))
    handles.append(qkvParam.register_forward_hook(val0_forward_hook))
    #------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------

    visOutProjParam = vision_blocks[VisionLayerTrack].attn.proj
    handles.append(visOutProjParam.register_forward_pre_hook(visOutProj_pre_hook))
    handles.append(visOutProjParam.register_forward_hook(visOutProj_forward_hook))

    #------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------

    # MODIFIED vs Gemma: FC1/FC2 -> visGate/visUp/visDown (SwiGLU MLP)
    visGateParam = vision_blocks[VisionLayerTrack].mlp.gate_proj
    handles.append(visGateParam.register_forward_pre_hook(visGate_pre_hook))
    handles.append(visGateParam.register_forward_hook(visGate_forward_hook))
    #------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------

    visUpParam = vision_blocks[VisionLayerTrack].mlp.up_proj
    handles.append(visUpParam.register_forward_pre_hook(visUp_pre_hook))
    handles.append(visUpParam.register_forward_hook(visUp_forward_hook))
    #------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------

    visDownParam = vision_blocks[VisionLayerTrack].mlp.down_proj
    handles.append(visDownParam.register_forward_pre_hook(visDown_pre_hook))
    handles.append(visDownParam.register_forward_hook(visDown_forward_hook))

    allTopRightSingularVectorsVis = [getTopRightSingularVectorForVisionQKV(qkvParam, "q", d_model, d_head, num_heads),
                                getTopRightSingularVectorForVisionQKV(qkvParam, "k", d_model, d_head, num_heads),
                                getTopRightSingularVectorForVisionQKV(qkvParam, "v", d_model, d_head, num_heads),
                                getTopRightSingularVector(visOutProjParam),
                                getTopRightSingularVector(visGateParam),
                                getTopRightSingularVector(visUpParam),
                                getTopRightSingularVector(visDownParam)]

    return allTopRightSingularVectorsVis, handles



def getallTopRightSingularVectorsLan(language_layers, LanLayerTrack, num_headsT, d_modelT, num_kv_headsT):
    handles = []

    qryLanParam = language_layers[LanLayerTrack].self_attn.q_proj
    handles.append(qryLanParam.register_forward_pre_hook(qryLan_pre_hook))
    handles.append(qryLanParam.register_forward_hook(qryLan_forward_hook))


    keyLanParam = language_layers[LanLayerTrack].self_attn.k_proj
    handles.append(keyLanParam.register_forward_pre_hook(keyLan_pre_hook))
    handles.append(keyLanParam.register_forward_hook(keyLan_forward_hook))

    valLanParam = language_layers[LanLayerTrack].self_attn.v_proj
    handles.append(valLanParam.register_forward_pre_hook(valLan_pre_hook))
    handles.append(valLanParam.register_forward_hook(valLan_forward_hook))


    gate_proj = language_layers[LanLayerTrack].mlp.gate_proj # layer 0 doing great
    handles.append(gate_proj.register_forward_pre_hook(gate_proj_pre_hook))
    handles.append(gate_proj.register_forward_hook(gate_proj_forward_hook))

    up_proj = language_layers[LanLayerTrack].mlp.up_proj # layer 0 doing great
    handles.append(up_proj.register_forward_pre_hook(up_proj_pre_hook))
    handles.append(up_proj.register_forward_hook(up_proj_forward_hook))


    down_proj = language_layers[LanLayerTrack].mlp.down_proj # layer 0 doing great
    handles.append(down_proj.register_forward_pre_hook(down_proj_pre_hook))
    handles.append(down_proj.register_forward_hook(down_proj_forward_hook))

    allTopRightSingularVectorsLan = [getTopRightSingularVectorForLanAttentioHeads(qryLanParam, num_headsT, d_modelT),
                                getTopRightSingularVectorForLanAttentioHeads(keyLanParam, num_kv_headsT, d_modelT),
                                getTopRightSingularVectorForLanAttentioHeads(valLanParam, num_kv_headsT, d_modelT),
                                getTopRightSingularVector(gate_proj),
                                getTopRightSingularVector(up_proj),
                                getTopRightSingularVector(down_proj)]

    return allTopRightSingularVectorsLan, handles


def getallTopRightSingularVectorsMultiMod(vision_module):
    handles = []
    MulModProjParam = vision_module.merger.mlp[2]
    handles.append(MulModProjParam.register_forward_pre_hook(MulModProj_pre_hook))
    handles.append(MulModProjParam.register_forward_hook(MulModProj_forward_hook))

    allTopRightSingularVectorsMultiMod = [getTopRightSingularVector(MulModProjParam)]
    return allTopRightSingularVectorsMultiMod, handles


def _split_batched_coeffs(coeffs, B, token_dim):
    """
    OPTIMIZATION (note #5): splits a single batched coefficients tensor
    (produced by running B images through one forward pass) back into B
    per-sample tensors, in the same order the images were stacked in. Safe
    because: (a) the vision tower keeps each image's tokens in one
    contiguous block (verified against `get_vision_window_index` -- the
    per-image window permutation is offset per image and never crosses
    image boundaries), and (b) the language model's batch dimension is
    flattened batch-major by `_drop_batch_dim_if_present`, so sample i's
    tokens are exactly the i-th contiguous chunk.
    """
    total = coeffs.shape[token_dim]
    assert total % B == 0, f"token dim {total} not divisible by batch size {B}"
    per_sample = total // B
    return list(torch.split(coeffs, per_sample, dim=token_dim))


def _build_batched_perturbations(x_orig01, epsilon, num_random_varieties, device):
    """
    Builds the B = num_random_varieties images (1 "original" with delta=0,
    followed by (B-1) "weak adversary" images with iid Gaussian delta) that
    the original script fed through the model one at a time. The random
    draws are generated with the exact same sequential `torch.randn_like`
    calls, in the exact same order, as the original per-sample loop -- so
    the perturbations themselves are identical to before; only the
    forward pass over all of them is batched.
    """
    deltas = [torch.zeros_like(x_orig01)]
    for _ in range(1, num_random_varieties):
        deltas.append(torch.randn_like(x_orig01) * epsilon)

    x_variants = []
    for delta in deltas:
        x_adv01 = (x_orig01 + delta).clamp(0.0, 1.0)
        x_adv01 = torch.max(torch.min(x_adv01, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)
        x_variants.append(x_adv01)

    return torch.cat(x_variants, dim=0)  # (B,3,H,W)


def run_batched_alignment_analysis(
    model,
    processor,
    template_inputs,
    x_orig01,
    epsilon: float,
    device,
    attackMode: str,
    LayerTrack: int,
    allTopRightSingularVectors,
    num_random_varieties: int,
    vision_module,
    language_module,
):
    """
    Replaces the original per-sample `adam_attack_original_space` loop.

    For a given tracked layer, builds all `num_random_varieties` perturbed
    images (1 original + (N-1) weak-adversary draws, see
    `_build_batched_perturbations`) and runs them through the model in a
    SINGLE batched forward pass instead of N sequential ones (note #5),
    using the narrowest possible sub-module for each attack mode (note #4):
      - "vis"/"mulmod": only `vision_module(...)` runs (language model +
        LM head are skipped entirely -- neither hook path touches them).
      - "lan": the base multimodal model runs (LM head skipped), with the
        language decoder's layer list temporarily truncated to
        `LayerTrack + 1` layers (layers after the tracked one cannot affect
        any hook-captured activation at or before `LayerTrack`).

    Returns a list of length `num_random_varieties`, where entry i is the
    list of per-hook coefficient tensors for sample i (index 0 = original,
    indices 1..N-1 = weak adversary draws) -- i.e. exactly the
    `FlattenedAlignmentDistributions*` lists the original per-call loop
    produced, just computed all at once.
    """
    B = num_random_varieties
    x_batch01 = _build_batched_perturbations(x_orig01, epsilon, B, device).to(device)

    pv_b, grid_b = qwen_preprocess_differentiable_batch(x_batch01, processor)
    grid_row = grid_b[0].tolist()
    grid_t, grid_h, grid_w = grid_row
    n_patches_raw_per_image = int(grid_t * grid_h * grid_w)

    with torch.inference_mode():
        if attackMode in ("vis", "mulmod"):
            # Only the vision tower matters for these hooks -- skip the
            # language model + LM head entirely.
            if attackMode == "vis":
                # Layers strictly after LayerTrack cannot influence any
                # hook captured at LayerTrack, so skip running them.
                with _truncated_module_list(vision_module, "blocks", LayerTrack + 1):
                    vision_module(pv_b.type(vision_module.dtype), grid_thw=grid_b)
            else:
                # "mulmod" hooks the merger, which runs after ALL blocks.
                vision_module(pv_b.type(vision_module.dtype), grid_thw=grid_b)

        elif attackMode == "lan":
            input_ids_b = template_inputs["input_ids"].repeat(B, 1)
            attention_mask_b = template_inputs.get("attention_mask")
            if attention_mask_b is not None:
                attention_mask_b = attention_mask_b.repeat(B, 1)

            with _truncated_module_list(language_module, "layers", LayerTrack + 1):
                model.model(
                    input_ids=input_ids_b,
                    attention_mask=attention_mask_b,
                    pixel_values=pv_b.type(vision_module.dtype),
                    image_grid_thw=grid_b,
                    use_cache=False,
                )
        else:
            raise ValueError(f"Unknown attackMode: {attackMode}")

        seq_len = int(template_inputs["input_ids"].shape[1])

        FlattenedAlignmentDistributionsAllSamples = [[] for _ in range(B)]

        def _collect(InputToLayer, sv, is_attention_head, token_count_per_sample):
            if is_attention_head:
                _, coeffs = getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, sv)
                chunks = _split_batched_coeffs(coeffs, B, token_dim=1)
            else:
                _, coeffs = getMeanAlignmentWithTopRightSingularVector(InputToLayer, sv)
                chunks = _split_batched_coeffs(coeffs, B, token_dim=0)
            for i in range(B):
                FlattenedAlignmentDistributionsAllSamples[i].append(chunks[i])

        if attackMode == "vis":
            _collect(qry0_inputs.get("qry0_in"), allTopRightSingularVectors[0], True, n_patches_raw_per_image)
            _collect(key0_inputs.get("key0_in"), allTopRightSingularVectors[1], True, n_patches_raw_per_image)
            _collect(val0_inputs.get("val0_in"), allTopRightSingularVectors[2], True, n_patches_raw_per_image)
            _collect(visOutProj_inputs.get("visOutProj_in"), allTopRightSingularVectors[3], False, n_patches_raw_per_image)
            _collect(visGate_inputs.get("visGate_in"), allTopRightSingularVectors[4], False, n_patches_raw_per_image)
            _collect(visUp_inputs.get("visUp_in"), allTopRightSingularVectors[5], False, n_patches_raw_per_image)
            _collect(visDown_inputs.get("visDown_in"), allTopRightSingularVectors[6], False, n_patches_raw_per_image)

        elif attackMode == "lan":
            _collect(qryLan_inputs.get("qryLan_in"), allTopRightSingularVectors[0], True, seq_len)
            _collect(keyLan_inputs.get("keyLan_in"), allTopRightSingularVectors[1], True, seq_len)
            _collect(valLan_inputs.get("valLan_in"), allTopRightSingularVectors[2], True, seq_len)
            _collect(GateLayerInputs.get("gate_in"), allTopRightSingularVectors[3], False, seq_len)
            _collect(UpLayerInputs.get("up_in"), allTopRightSingularVectors[4], False, seq_len)
            _collect(DownLayer_inputs.get("down_in"), allTopRightSingularVectors[5], False, seq_len)

        elif attackMode == "mulmod":
            _collect(MulModProj_inputs.get("MulModProj_in"), allTopRightSingularVectors[0], False, None)

    return FlattenedAlignmentDistributionsAllSamples


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL ORIGINAL-image-space adversarial attack (no squeeze)")
    parser.add_argument("--attck_type", type=str, default="bsa",
                        help="bsa | bsa_flat | bsa_flat_lan | bsa_flat_vis")
    parser.add_argument("--desired_norm_l_inf", type=float, default=0.03,
                        help="epsilon L_inf in ORIGINAL pixel space [0..1]. Try 0.01~0.08")
    parser.add_argument("--learningRate", type=float, default=1e-3,
                        help="Adam learning rate")
    parser.add_argument("--num_steps", type=int, default=2000,
                        help="Number of Adam steps")
    #parser.add_argument("--attackSample", type=str, default="nature",
                    #help="which sample")
    parser.add_argument("--AttackStartLayer", type=int, default=0,
                        help="From which layer do you start attack")
    parser.add_argument("--numLayerstAtAtime", type=int, default=2,
                        help="Number of layers taken at a time to attack")

    parser.add_argument("--VisionLayerTrack", type=int, default=2,
                        help="whcih vision layer you want to talk")
    parser.add_argument("--LanLayerTrack", type=int, default=2,
                        help="whcih language layer you want to talk")
    parser.add_argument("--kthSingVec", type=int, default=0,
                    help="Amonhg the k singular vectors which one do you wanty")
    parser.add_argument("--attackMode", type=str, default="lan",
                    help="Which layer were attacked vis or lan")



    args = parser.parse_args()

    attck_type = args.attck_type
    epsilon = float(args.desired_norm_l_inf)
    lr = float(args.learningRate)
    num_steps = int(args.num_steps)
    #attackSample = str(args.attackSample)
    AttackStartLayer = int(args.AttackStartLayer)
    numLayerstAtAtime = int(args.numLayerstAtAtime)

    VisionLayerTrack = int(args.VisionLayerTrack)
    LanLayerTrack = int(args.LanLayerTrack)

    kthSingVec = int(args.kthSingVec)
    attackMode = str(args.attackMode)


    MODEL_PATH = "../illcond/QwenAttack/Qwen2.5-VL-7B-Instruct"
    QUESTION = "What is shown in this image?"
    MAX_NEW_TOKENS = 128

    point_labels = [
    "query proj", "key proj", "value proj", "att output\nproj",
    "MLP gate\n(vis)", "MLP up\n(vis)", "MLP down\n(vis)", "Vis-to-lan\nproj", "query proj\n",
    "key proj\n", "value proj\n", "MLP gate\nproj", "MLP up\nproj",
    "MLP down\nproj"
    ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"device={device}, dtype={dtype}")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)

    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=dtype,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    #----------------------attention head hyper parameters extraction------------------------
    vision_module, vision_blocks = get_vision_module_and_blocks(model)
    language_module, language_layers = get_language_module_and_layers(model)

    vision_cfg = _resolve_vision_config(model)
    text_cfg = _resolve_text_config(model)

    num_heads = getattr(vision_cfg, "num_heads", 16)
    d_model = getattr(vision_cfg, "hidden_size", vision_blocks[0].attn.qkv.weight.shape[1])
    d_head = d_model // num_heads
    #----------------------attention head hyper parameters extraction------------------------
    d_modelT = getattr(text_cfg, "hidden_size")
    num_headsT = getattr(text_cfg, "num_attention_heads")

    num_kv_headsT = getattr(text_cfg, "num_key_value_heads", num_headsT)





    attackSample = 1
    IMAGE_PATH = f"../interpretAttacks/llava_attack/dataSamplesForQuant/{attackSample}.JPEG"
    pil = Image.open(IMAGE_PATH).convert("RGB")
    x_orig01 = pil_to_tensor01(pil).to(device)
    template_inputs = build_template_inputs(processor, QUESTION, pil, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    NumRandomVarieties = 2

    def ImportantSingularVectorSampling(attackMode, numLayersInBlock, point_labels):

        for LayerTrack in range(numLayersInBlock):

            with torch.no_grad():

                if attackMode == "vis":
                    allTopRightSingularVectorsVis, hook_handles = getAllallTopRightSingularVectorsVis(vision_blocks, LayerTrack, d_model, d_head, num_heads)
                    point_labels_ = point_labels[:7]
                    whichRightSingularVectors = allTopRightSingularVectorsVis

                if attackMode == "lan":
                    allTopRightSingularVectorsLan, hook_handles = getallTopRightSingularVectorsLan(language_layers, LayerTrack, num_headsT, d_modelT, num_kv_headsT)
                    point_labels_ = point_labels[8:]
                    whichRightSingularVectors = allTopRightSingularVectorsLan

                if attackMode == "mulmod":
                    allTopRightSingularVectorsMultiMod, hook_handles = getallTopRightSingularVectorsMultiMod(vision_module)
                    point_labels_ = point_labels[7]
                    whichRightSingularVectors = allTopRightSingularVectorsMultiMod

            try:
                # OPTIMIZATION (notes #1, #2, #3, #4, #5): a single batched
                # forward pass (through the narrowest sub-module the mode
                # needs) replaces the original's 10 sequential
                # forward-pass-pairs (20 full-model forward passes) per
                # layer.
                FlattenedAlignmentDistributionsAllSamples = run_batched_alignment_analysis(
                    model=model,
                    processor=processor,
                    template_inputs=template_inputs,
                    x_orig01=x_orig01,
                    epsilon=epsilon,
                    device=device,
                    attackMode=attackMode,
                    LayerTrack=LayerTrack,
                    allTopRightSingularVectors=whichRightSingularVectors,
                    num_random_varieties=NumRandomVarieties,
                    vision_module=vision_module,
                    language_module=language_module,
                )
            finally:
                # OPTIMIZATION (note #6): remove this layer's hooks now so
                # they don't keep firing (uselessly) on every later layer's
                # forward pass for the rest of the sweep.
                for h in hook_handles:
                    h.remove()

            FlattenedAlignmentDistributionsOriginal = FlattenedAlignmentDistributionsAllSamples[0]

            AggregationOverFlattenedAlignmentDistributionsOriginal = []
            averagedAggregationOverFlattenedAlignmentDistributionsOriginal = []
            averagedAggregationOverFlattenedAlignmentDistributionsOriginalSTD = []

            for rndVar in range(1, NumRandomVarieties):
                FlattenedAlignmentDistributionsWeak = FlattenedAlignmentDistributionsAllSamples[rndVar]

                DiffWeakThisSample = []
                for i in range(len(FlattenedAlignmentDistributionsOriginal)):
                    orig_c = FlattenedAlignmentDistributionsOriginal[i]
                    weak_c = FlattenedAlignmentDistributionsWeak[i]

                    token_dim = 0 if orig_c.dim() == 2 else 1
                    diffWeak = (orig_c - weak_c).pow(2).mean(dim=token_dim).sqrt().flatten()

                    DiffWeakThisSample.append(diffWeak)

                AggregationOverFlattenedAlignmentDistributionsOriginal.append(DiffWeakThisSample)

                averagedAggregationOverFlattenedAlignmentDistributionsOriginal = [
                    torch.stack(elements).mean(dim=0)
                    for elements in zip(*AggregationOverFlattenedAlignmentDistributionsOriginal)
                ]

                averagedAggregationOverFlattenedAlignmentDistributionsOriginalSTD = [
                    torch.stack(elements).std(dim=0)
                    for elements in zip(*AggregationOverFlattenedAlignmentDistributionsOriginal)
                ]


            for i in range(len(averagedAggregationOverFlattenedAlignmentDistributionsOriginal)):

                weak = averagedAggregationOverFlattenedAlignmentDistributionsOriginal[i].detach().to(torch.float32).cpu().numpy()
                weak_mean = np.mean(weak)
                weak_norm = weak / weak_mean
                print("i",i)
                print("point_labels[i]", point_labels_[i])
                label = point_labels_[i].replace("\n", " ")

                L = len(weak)
                x = np.arange(L)
                weak_std = averagedAggregationOverFlattenedAlignmentDistributionsOriginalSTD[i].detach().to(torch.float32).cpu().numpy()
                weak_std_norm = weak_std / weak_mean
                print(f" {attackMode} layer track", LayerTrack)
                print("weak_norm.shape", weak_norm.shape)






    numLayersInBlock = 32 #32
    attackMode = "vis"
    ImportantSingularVectorSampling(attackMode, numLayersInBlock, point_labels)

    numLayersInBlock = 1
    attackMode = "mulmod"
    ImportantSingularVectorSampling(attackMode, numLayersInBlock, point_labels)

    numLayersInBlock = 27
    attackMode = "lan"
    ImportantSingularVectorSampling(attackMode, numLayersInBlock, point_labels)


if __name__ == "__main__":
    main()