

'''

Gemma3 port of qwen/Qwen2p5FullDistancesOverlap.py. Same weak-vs-strong
adversary L2-distance-overlap analysis, applied to Gemma3 instead of
Qwen2.5-VL. Model loading, image preprocessing, hook points/module getters,
per-head SVD helpers, and the BSA adversarial-noise-loading convention are
all taken from
gemma_attack/RealSpectralSubSpaceAlignmentPostAttackExaminerFullDistancesOverlap.py
(the existing Gemma3 post-attack examiner), while the analysis/plotting logic
itself (raw un-reduced coeffs, robust token-axis reduction, three plot types
per tracked point, CSV summary logging) mirrors
qwen/Qwen2p5FullDistancesOverlap.py.

For a given (VisionLayerTrack, LanLayerTrack, attackMode) this script:

  1. Computes the FULL right/left singular-vector bases (per attention head
     where relevant) of every tracked linear operator at that layer: vision
     q/k/v/out_proj + MLP fc1/fc2, the multi_modal_projector merger, and
     language q/k/v + MLP gate/up/down.

  2. For each of ~49 dataset samples, runs THREE forward passes against the
     SAME image: (a) the trained/loaded BSA adversarial perturbation
     ("strong" adversary), (b) zero perturbation ("original"), and (c) a
     Gaussian perturbation at the same L_inf epsilon ("weak" adversary),
     capturing the raw (pre-reduction) per-token alignment coefficients
     against each tracked operator's singular-vector basis.

  3. Computes, per tracked operator and per singular-vector index, an
     RMS-style L2 distance between (original vs strong) and (original vs
     weak) coefficients, reducing over the TOKEN axis (robustly detected via
     `token_dim = 0 if coeffs.dim() == 2 else 1`, since the attention-head
     coeffs have shape (heads, N, k) while the plain-Linear coeffs have shape
     (N, k)).

  4. Averages these L2-distance curves across all samples and plots, for
     every tracked point: (a) a raw overlaid bar chart (weak vs strong), (b)
     a mean +/- std band chart, and (c) the same band chart normalized by
     each curve's own mean. A running CSV logs the overall weak-vs-strong L2
     gap per point/layer, so a full LanLayerTrack/VisionLayerTrack sweep
     (see example commands below) can be compared to spot which layer shows
     the largest weak-vs-strong separation.

ARCHITECTURE DIFFERENCES vs Qwen2.5-VL that this port had to account for
(all taken directly from
RealSpectralSubSpaceAlignmentPostAttackExaminerFullDistancesOverlap.py):

  - Vision tower (SigLIP, "vision_tower.vision_model.encoder.layers.N"):
    query/key/value are three SEPARATE nn.Linear modules
    (self_attn.q_proj/k_proj/v_proj), NOT a fused attn.qkv like Qwen2.5-VL,
    so each is hooked and SVD'd directly (no row-slicing needed). Attention
    output proj is self_attn.out_proj (not attn.proj). The vision MLP is a
    plain 2-layer GELU FFN (mlp.fc1 / mlp.fc2), NOT a 3-way SwiGLU
    gate/up/down block like Qwen's vision MLP -- so there are only 6 tracked
    vision points here (query, key, value, att output proj, fc1, fc2)
    instead of Qwen's 7 (extra "MLP up (vis)" entry).

  - Vision-to-language bridge: Gemma3's multi_modal_projector has no
    nn.Linear sub-module (unlike Qwen's visual.merger.mlp, a plain
    nn.Sequential(Linear, GELU, Linear)). Instead it stores a single raw
    parameter, mm_input_projection_weight, shape (vision_hidden=1152,
    text_hidden=2560) -- i.e. (in_features, out_features), the TRANSPOSE of
    nn.Linear's (out,in) convention. getTopRightSingularVectorForMMproj /
    getTopLeftSingularVectorForMMproj therefore SVD
    mm_input_projection_weight.T instead of calling the generic
    getTopRightSingularVector/getTopLeftSingularVector used everywhere else.

  - Language decoder ("language_model.model.layers.N"): structurally the
    same SwiGLU MLP and GQA attention as Qwen, but Gemma3's language model
    ALSO uses GQA (num_attention_heads=8, num_key_value_heads=4). The
    original Gemma examiner this file is ported from reused a single
    num_headsT constant for query/key/value alike when reshaping into
    per-head blocks, which is fine for query (2048 // 8 = 256, the real
    head_dim) but silently WRONG for key/value (1024 // 8 = 128 "heads" of
    the wrong size, instead of the true 4 heads of 256). This port fixes
    that by passing num_headsT for query and num_kv_headsT for key/value
    into getTopRightSingularVectorForLanAttentioHeads /
    getTopLeftSingularVectorForLanAttentioHeads -- the same GQA correctness
    fix qwen/Qwen2p5FullDistancesOverlap.py already applies for Qwen2.5-VL's
    own GQA language model (see its "MODIFIED vs Gemma" comment there).

  - BSA adversarial-noise loading: Gemma3's BSA trainer names its output
    files with towardsNull / lanMLP / visMLP / balancingAlpha tags (no Qwen
    equivalent exists), so the adv_noise_path construction is copied
    VERBATIM from
    RealSpectralSubSpaceAlignmentPostAttackExaminerFullDistancesOverlap.py
    rather than reusing Qwen's much simpler BSA filename convention.

Plots and the CSV summary are written under gemma_attack/ (created at
runtime if missing):
  gemma_attack/OverlapDistancesAvg/                     (raw bar charts + weak_vs_strong_l2_summary.csv)
  gemma_attack/OverlapDistancesAvgStdBands/             (mean +/- std band charts)
  gemma_attack/OverlapDistancesAvgStdBandsNormalized/   (mean +/- std band charts, normalized by mean)

Example runs:

export CUDA_VISIBLE_DEVICES=2
conda deactivate
cd spectralShift/
conda activate gemma3
export PYTHONNOUSERSITE=1
for StudyLayer in $(seq 0 33); do
    python gemma_attack/Gemma3FullDistancesOverlap.py --attck_type saa_BSAexp --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --VisionLayerTrack 0 --LanLayerTrack $StudyLayer --kthSingVec -10 --attackMode lan
done

for StudyLayer in $(seq 0 26); do
    python gemma_attack/Gemma3FullDistancesOverlap.py --attck_type saa_BSAexp --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --VisionLayerTrack $StudyLayer --LanLayerTrack 0 --kthSingVec 10 --attackMode vis
done

'''




import os
# MODIFIED vs Qwen: this must be set before any CUDA/cuBLAS context is
# created for deterministic matmuls to actually take effect, so it has to
# come immediately after `import os`, before torch is imported (ported
# verbatim from qwen/Qwen2p5FullDistancesOverlap.py).
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import sys
import csv
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import matplotlib.pyplot as plt

# ----------------------------
# Reproducibility
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
# Losses: GRILL + OA (kept for compatibility, unused by this examiner)
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
# Differentiable preprocessing (Gemma3)
# Ported verbatim from
# RealSpectralSubSpaceAlignmentPostAttackExaminerFullDistancesOverlap.py /
# gemma_attack/gemma3Inference.py. Gemma3's image processor does a fixed-size
# resize + center-crop (no dynamic-resolution patch-grid like Qwen2.5-VL), so
# there is no image_grid_thw here.
# ----------------------------
def _get_target_hw(image_processor):
    """
    Try to infer model target H,W from HF image_processor.
    We handle common formats:
      - ip.size = {"height": H, "width": W}
      - ip.size = {"shortest_edge": S}
      - ip.size = S (int)
      - ip.crop_size likewise
    """
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
        # fallback for many gemma/vlm configs
        target_h = target_w = 896

    return int(target_h), int(target_w)


def resize_keep_aspect_center_crop(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    Differentiable:
      - scale so that resized image >= target in both dims
      - center crop to (target_h, target_w)
    """
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


def gemma_preprocess_differentiable(x01: torch.Tensor, processor) -> torch.Tensor:
    """
    Differentiable approximation of the processor's image pipeline.
    Produces pixel_values like the processor would (shape 1x3xH'xW').
    """
    ip = processor.image_processor
    th, tw = _get_target_hw(ip)
    x = resize_keep_aspect_center_crop(x01, th, tw)
    x = normalize_like_processor(x, ip)
    return x


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
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Pass an image ONCE so the processor inserts the correct special image token(s)
    template = processor(text=[prompt], images=[pil_image], return_tensors="pt")
    template = {k: v.to(device) if torch.is_tensor(v) else v for k, v in template.items()}
    return template


# ----------------------------
# Generation helper (uses template, swaps pixel_values)
# MODIFIED vs Qwen: Gemma3 needs only pixel_values (no image_grid_thw).
# ----------------------------
def run_generation_with_pixel_values(model, processor, template_inputs, pixel_values, max_new_tokens=128):
    model.eval()
    inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
    inputs["pixel_values"] = pixel_values

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
    Ported from qwen/Qwen2p5FullDistancesOverlap.py for structural parity.
    For Gemma3 this always simplifies to plain `t[0]`: both the SigLIP
    vision tower and the language model keep a leading batch dimension of
    size 1 for every hooked tensor (unlike Qwen2.5-VL's vision transformer,
    which runs on a single FLAT (total_tokens, hidden_dim) tensor with no
    batch dimension at all). Kept as a helper (rather than hardcoding
    `InputToLayer[0]` the way the original Gemma examiner did) so the same
    alignment functions below are copy-identical to the Qwen version.
    """
    if t.dim() == 3 and t.shape[0] == 1:
        return t[0]
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

    # Mean across tokens
    mean_energy = per_token_energy.mean().item()

    # Return the raw (un-reduced) coeffs instead of coeffs.mean(dim=1).
    # We need the pre-reduction coeffs so that, at the call site, we can
    # compute an L2 difference against another (original/weak/strong)
    # coeffs tensor BEFORE doing the token-wise averaging.
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
    H = _drop_batch_dim_if_present(InputToLayer)          # (N, d_model)
    V = topRightSingularVector   # (num_heads, k, d_model)
    V = V.to(H)
    H_hat = F.normalize(H, dim=1)        # (N, d_model)
    V_hat = F.normalize(V, dim=2)        # (num_heads, k, d_model)

    coeffs = torch.einsum('nd,hkd->hnk', H_hat, V_hat) # dot product between all the bottom-k singular vectors from all the heads on the weight matrix, and all the input tokens.

    energy = (coeffs ** 2).sum(dim=2) # per-token, per-head subspace energy.
    mean_energy_per_head = energy.mean(dim=1)  # (num_heads,)
    mean_energy_all = mean_energy_per_head.mean().item()

    # Return the raw (un-reduced) coeffs (shape h,n,k) instead of
    # coeffs.mean(dim=1). Same reasoning as above: we need the pre-reduction
    # tensor to compute an L2 difference against another coeffs tensor first.
    # NOTE: this helper is fully generic in num_heads (it just reads V's own
    # shape), so it works unchanged for Gemma3's vision heads (MHA, 16
    # heads) AND for the language-model query heads (8) as well as the GQA
    # key/value heads (4) -- no modification needed here.
    return mean_energy_all, coeffs


def getMeanAlignmentWithAttentionHeadTopLeftSingularVector(OutputOfLayer, num_headsT, topLeftSingularVector):

    H = _drop_batch_dim_if_present(OutputOfLayer)          # (N, d_model)
    V = topLeftSingularVector   # (num_heads, d_model)
    V = V.to(H)
    H_hat = F.normalize(H, dim=1)        # (N, d_model)
    V_hat = F.normalize(V, dim=1)        # (num_heads, d_model)

    H_hat = H_hat.view(H_hat.shape[0], num_headsT, -1)
    dots = H_hat @ V_hat.T               # (N, num_heads)
    mean_abs_value = dots.abs().mean().item()

    return mean_abs_value


def getMeanAlignmentWithLanAttentionHeadTopRightSingularVector(InputToLayer, topRightSingularVector):
    H = _drop_batch_dim_if_present(InputToLayer)          # (N, d_model)
    V = topRightSingularVector   # (num_heads, d_model)
    V = V.to(H)
    H_hat = F.normalize(H, dim=1)        # (N, d_model)
    V_hat = F.normalize(V, dim=1)        # (num_heads, d_model)
    dots = H_hat @ V_hat.T               # (N, num_heads)
    mean_abs_value = dots.abs().mean().item()

    return mean_abs_value


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Robust getters for the vision blocks / language layers / vision-to-language
# merger, in the same defensive hasattr() style
# qwen/Qwen2p5FullDistancesOverlap.py uses for Qwen2.5-VL. Gemma3's own
# post-attack examiner accessed these paths directly (no fallback probing);
# these getters keep that same known-good path as the primary branch.
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
def get_vision_module_and_blocks(model):
    if hasattr(model, "vision_tower") and hasattr(model.vision_tower, "vision_model") and hasattr(model.vision_tower.vision_model, "encoder"):
        return model.vision_tower.vision_model, model.vision_tower.vision_model.encoder.layers
    else:
        raise RuntimeError("Could not find Gemma3 vision encoder layers.")


def get_language_module_and_layers(model):
    if hasattr(model, "language_model") and hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
        return model.language_model.model, model.language_model.model.layers
    elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model, model.language_model.layers
    else:
        raise RuntimeError("Could not find Gemma3 language-model layers.")


def get_multi_modal_projector(model):
    if hasattr(model, "multi_modal_projector"):
        return model.multi_modal_projector
    else:
        raise RuntimeError("Could not find Gemma3 multi_modal_projector module.")


def get_vision_attn_head_config(model, vision_module, vision_blocks):
    """
    Same extraction Gemma3's own examiner uses: read num_heads off the
    first vision block's self_attn, and read d_model off the vision
    embeddings' 4D conv weight (patch_embedding.weight), which SigLIP
    otherwise doesn't expose directly on the attention module.
    """
    layer0 = vision_blocks[0]
    num_heads = layer0.self_attn.num_heads
    d_model = None
    for name, param in vision_module.embeddings.named_parameters():
        if param.dim() == 4:
            d_model = param.shape[0]
            break
    if d_model is None:
        # Fallback: derive from q_proj's own weight shape.
        d_model = layer0.self_attn.q_proj.weight.shape[1]
    d_head = d_model // num_heads
    return num_heads, d_head, d_model


def get_language_attn_head_config(model):
    text_cfg = model.language_model.config
    d_modelT = getattr(text_cfg, "hidden_size")
    num_headsT = getattr(text_cfg, "num_attention_heads")
    # MODIFIED vs the original Gemma examiner (necessary correctness fix,
    # ported from qwen/Qwen2p5FullDistancesOverlap.py's own GQA fix):
    # Gemma3's language model uses Grouped-Query Attention just like
    # Qwen2.5-7B-Instruct's (num_key_value_heads < num_attention_heads).
    # The original examiner reused num_headsT for k_proj/v_proj too, which
    # only produces a VALID reshape (1024 // 8 = 128), not a CORRECT one --
    # it silently treats key/value as 8 heads of dim 128 instead of the
    # true 4 heads of dim 256. num_kv_headsT must be used for k_proj/v_proj
    # so the per-head reshape matches Gemma3's actual attention geometry.
    num_kv_headsT = getattr(text_cfg, "num_key_value_heads", num_headsT)
    return num_headsT, num_kv_headsT, d_modelT


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

# Gemma3's vision MLP is a plain 2-layer FC (fc1/fc2), NOT a 3-projection
# SwiGLU block like Qwen2.5-VL's vision MLP -- so there are only two vision
# MLP hooks here (FC1/FC2), unlike Qwen's three (visGate/visUp/visDown).
FC1_inputs = {}
def FC1_pre_hook(module, inputs):
    FC1_inputs["FC1_in"] = inputs[0]
FC1_outputs = {}
def FC1_forward_hook(module, inputs, output):
    FC1_outputs["FC1_out"] = output
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
FC2_inputs = {}
def FC2_pre_hook(module, inputs):
    FC2_inputs["FC2_in"] = inputs[0]
FC2_outputs = {}
def FC2_forward_hook(module, inputs, output):
    FC2_outputs["FC2_out"] = output
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

# ----------------------------
# ORIGINAL-SPACE Adam attack (here: single forward pass against a *loaded*
# perturbation -- this file is a post-attack examiner, not a trainer)
# ----------------------------
def adam_attack_original_space(
    model,
    processor,
    template_inputs,
    x_orig01,               # (1,3,H0,W0) in [0,1]
    attck_type: str,
    num_steps: int,
    lr: float,
    epsilon: float,         # L_inf bound in ORIGINAL pixel space [0,1]
    device,
    #save_conv_path: str,
    AttackStartLayer: int,
    numLayerstAtAtime: int,
    allTopRightSingularVectors,
    best_delta
):

    x_orig01 = x_orig01.detach().to(device)


    delta = best_delta
    delta.requires_grad_(True)


    best_delta = delta.detach().clone()

    model.eval()

    with torch.no_grad():
        pv_clean_fixed = gemma_preprocess_differentiable(x_orig01, processor)

        clean_inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
        clean_inputs["pixel_values"] = pv_clean_fixed
        clean_inputs["labels"] = template_inputs["input_ids"]
        clean_inputs["use_cache"] = False
        outputsN = model(**clean_inputs, output_hidden_states=True, return_dict=True)
        hiddStateLen = len(outputsN.hidden_states)

        startPos = AttackStartLayer
        endPos = startPos + numLayerstAtAtime
        if endPos > hiddStateLen:
            raise ValueError(
                f"endPos ({endPos}) exceeds number of hidden states ({hiddStateLen})"
            )

    adv_inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
    adv_inputs["labels"] = template_inputs["input_ids"]
    adv_inputs["use_cache"] = False
    RightSingularInputAlignmentWhole = []
    LeftSingularOutputStepAlignmentWhole = []

    x_adv01 = (x_orig01 + delta).clamp(0.0, 1.0)
    x_adv01 = torch.max(torch.min(x_adv01, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

    # preprocess adv (must be differentiable)
    pv_adv = gemma_preprocess_differentiable(x_adv01, processor)

    adv_inputs["pixel_values"] = pv_adv

    outputs = model(**adv_inputs, output_hidden_states=True, return_dict=True)

    RightSingularInputAlignment = []
    LeftSingularOutputStepAlignment = []
    AlignmentDistributions = []
    with torch.no_grad():

        InputToLayer = qry0_inputs.get("qry0_in")
        qry0_in_MAE, ProjDistrib = getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, allTopRightSingularVectors[0])
        RightSingularInputAlignment.append(qry0_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        InputToLayer = key0_inputs.get("key0_in")
        key0_in_MAE, ProjDistrib = getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, allTopRightSingularVectors[2])
        RightSingularInputAlignment.append(key0_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        InputToLayer = val0_inputs.get("val0_in")
        val0_in_MAE, ProjDistrib = getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, allTopRightSingularVectors[4])
        RightSingularInputAlignment.append(val0_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        InputToLayer = visOutProj_inputs.get("visOutProj_in")
        visOutProj_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[6])
        RightSingularInputAlignment.append(visOutProj_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        InputToLayer = FC1_inputs.get("FC1_in")
        FC1_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[8])
        RightSingularInputAlignment.append(FC1_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        InputToLayer = FC2_inputs.get("FC2_in")
        FC2_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[10])
        RightSingularInputAlignment.append(FC2_in_MAE)
        AlignmentDistributions.append(ProjDistrib)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        InputToLayer = MulModProj_inputs.get("MulModProj_in")
        MulModProj_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[12])
        RightSingularInputAlignment.append(MulModProj_in_MAE)
        AlignmentDistributions.append(ProjDistrib)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        InputToLayer = qryLan_inputs.get("qryLan_in")
        qryLan_in_MAE, ProjDistrib = getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, allTopRightSingularVectors[14])
        RightSingularInputAlignment.append(qryLan_in_MAE)
        AlignmentDistributions.append(ProjDistrib)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        InputToLayer = keyLan_inputs.get("keyLan_in")
        keyLan_in_MAE, ProjDistrib = getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, allTopRightSingularVectors[16])
        RightSingularInputAlignment.append(keyLan_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        InputToLayer = valLan_inputs.get("valLan_in")
        valLan_in_MAE, ProjDistrib = getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, allTopRightSingularVectors[18])
        RightSingularInputAlignment.append(valLan_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        InputToLayer = GateLayerInputs.get("gate_in")
        gate_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[20])
        RightSingularInputAlignment.append(gate_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        InputToLayer = UpLayerInputs.get("up_in")
        up_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[22])
        RightSingularInputAlignment.append(up_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        InputToLayer = DownLayer_inputs.get("down_in")
        down_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[24])
        RightSingularInputAlignment.append(down_in_MAE)
        AlignmentDistributions.append(ProjDistrib)


    RightSingularInputAlignment = np.array(RightSingularInputAlignment)

    FlattenedAlignmentDistributions = []
    for i in range(len(AlignmentDistributions)):
        # Kept un-flattened (raw coeffs, shape (N,k) or (h,n,k)) instead of
        # AlignmentDistributions[i].flatten(). We need the original
        # (pre-reduction) shape at the call site to compute an L2 difference
        # against another adversary's coeffs BEFORE doing the token-wise
        # averaging that used to happen inside the helper functions above.
        FlattenedAlignmentDistributions.append(AlignmentDistributions[i])

    with torch.no_grad():
        x_adv01_final = (x_orig01 + best_delta).clamp(0.0, 1.0)
        x_adv01_final = torch.max(torch.min(x_adv01_final, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

    return x_adv01_final, best_delta, RightSingularInputAlignment, FlattenedAlignmentDistributions


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Gemma3 ORIGINAL-image-space adversarial attack (no squeeze)")
    parser.add_argument("--attck_type", type=str, default="bsa",
                        help="bsa | bsa_flat | bsa_flat_lan | bsa_flat_vis")
    parser.add_argument("--desired_norm_l_inf", type=float, default=0.03,
                        help="epsilon L_inf in ORIGINAL pixel space [0..1]. Try 0.01~0.08")
    parser.add_argument("--learningRate", type=float, default=1e-3,
                        help="Adam learning rate")
    parser.add_argument("--num_steps", type=int, default=2000,
                        help="Number of Adam steps")
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
    AttackStartLayer = int(args.AttackStartLayer)
    numLayerstAtAtime = int(args.numLayerstAtAtime)

    VisionLayerTrack = int(args.VisionLayerTrack)
    LanLayerTrack = int(args.LanLayerTrack)

    kthSingVec = int(args.kthSingVec)
    attackMode = str(args.attackMode)


    MODEL_PATH = "../illcond/gemma_attack/Gemma3-4b"
    QUESTION = "What is shown in this image?"
    MAX_NEW_TOKENS = 128

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"device={device}, dtype={dtype}")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, padding_side="left")

    print("Loading model...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    #----------------------attention head hyper parameters extraction------------------------
    vision_module, vision_blocks = get_vision_module_and_blocks(model)
    language_module, language_layers = get_language_module_and_layers(model)

    num_heads, d_head, d_model = get_vision_attn_head_config(model, vision_module, vision_blocks)
    #----------------------attention head hyper parameters extraction------------------------


    def getTopRightSingularVector(down_proj):
        if kthSingVec<0:
            return torch.linalg.svd(down_proj.weight.to(torch.float32))[2][:]
        else:
            return torch.linalg.svd(down_proj.weight.to(torch.float32))[2][:]

    def getTopLeftSingularVector(down_proj):
        if kthSingVec<0:
            return torch.linalg.svd(down_proj.weight.to(torch.float32))[0][:]
        else:
            return torch.linalg.svd(down_proj.weight.to(torch.float32))[0][:]

    # MODIFIED vs Qwen: Gemma3's vision-to-language "merger" is
    # multi_modal_projector, which stores a raw (transposed)
    # mm_input_projection_weight parameter instead of a plain nn.Linear like
    # Qwen's visual.merger.mlp[2]. These two helpers SVD the transposed
    # weight so the resulting basis is in the same (out_features,
    # in_features) convention every other target uses.
    def getTopRightSingularVectorForMMproj(down_proj): # the difference is that this layer parameters were most likely transposed
        if kthSingVec<0:
            return torch.linalg.svd(down_proj.mm_input_projection_weight.T.to(torch.float32))[2][:]
        else:
            return torch.linalg.svd(down_proj.mm_input_projection_weight.T.to(torch.float32))[2][:]


    def getTopLeftSingularVectorForMMproj(down_proj): # the difference is that this layer parameters were most likely transposed
        if kthSingVec<0:
            return torch.linalg.svd(down_proj.mm_input_projection_weight.T.to(torch.float32))[0][:]
        else:
            return torch.linalg.svd(down_proj.mm_input_projection_weight.T.to(torch.float32))[0][:]

    def getTopRightSingularVectorForAttentioHeads(qryParam):
        param = qryParam.weight.to(torch.float32)
        param_heads = param.view(num_heads, d_head, d_model)
        query_vh_per_head = []
        for h in range(num_heads):
            Wh = param_heads[h]
            U, S, Vh = torch.linalg.svd(Wh.to(torch.float32) )
            if kthSingVec<0:
                query_vh_per_head.append(Vh[:])
            else:
                query_vh_per_head.append(Vh[:])
        query_vh_per_head = torch.stack(query_vh_per_head, 0)
        return query_vh_per_head


    def getTopLeftSingularVectorForAttentioHeads(qryParam):
        param = qryParam.weight.to(torch.float32)
        param_heads = param.view(num_heads, d_head, d_model)
        query_vh_per_head = []
        for h in range(num_heads):
            Wh = param_heads[h]
            U, S, Vh = torch.linalg.svd(Wh.to(torch.float32) )
            if kthSingVec<0:
                query_vh_per_head.append(U[:])
            else:
                query_vh_per_head.append(U[:])
        query_vh_per_head = torch.stack(query_vh_per_head, 0)
        return query_vh_per_head

    num_headsT, num_kv_headsT, d_modelT = get_language_attn_head_config(model)

    # MODIFIED vs the original Gemma examiner (necessary correctness fix,
    # ported from qwen/Qwen2p5FullDistancesOverlap.py's own GQA fix): this
    # helper now takes num_heads_local explicitly, so the call sites below
    # can pass num_headsT for query and num_kv_headsT for key/value instead
    # of reusing a single (wrong-for-GQA) num_headsT everywhere.
    def getTopRightSingularVectorForLanAttentioHeads(qryParam, num_heads_local):
        param = qryParam.weight.to(torch.float32)
        d_headT = param.shape[0] // num_heads_local
        param_heads = param.view(num_heads_local, d_headT, d_modelT)
        query_vh_per_head = []
        for h in range(num_heads_local):
            Wh = param_heads[h]
            U, S, Vh = torch.linalg.svd(Wh.to(torch.float32) )
            if kthSingVec<0:
                query_vh_per_head.append(Vh[:])
            else:
                query_vh_per_head.append(Vh[:])
        query_vh_per_head = torch.stack(query_vh_per_head, 0)
        return query_vh_per_head

    def getTopLeftSingularVectorForLanAttentioHeads(qryParam, num_heads_local):
        param = qryParam.weight.to(torch.float32)
        d_headT = param.shape[0] // num_heads_local
        param_heads = param.view(num_heads_local, d_headT, d_modelT)
        query_vh_per_head = []
        for h in range(num_heads_local):
            Wh = param_heads[h]
            U, S, Vh = torch.linalg.svd(Wh.to(torch.float32) )
            if kthSingVec<0:
                query_vh_per_head.append(U[:])
            else:
                query_vh_per_head.append(U[:])
        query_vh_per_head = torch.stack(query_vh_per_head, 0)
        return query_vh_per_head


    with torch.no_grad():

        # ------- vision hooks begin ----------------
        qryParam = vision_blocks[VisionLayerTrack].self_attn.q_proj
        hook_handle = qryParam.register_forward_pre_hook(qry0_pre_hook)
        hook_handle = qryParam.register_forward_hook(qry0_forward_hook)
        #------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------
        keyParam = vision_blocks[VisionLayerTrack].self_attn.k_proj
        hook_handle = keyParam.register_forward_pre_hook(key0_pre_hook)
        hook_handle = keyParam.register_forward_hook(key0_forward_hook)
        #------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------
        valParam = vision_blocks[VisionLayerTrack].self_attn.v_proj
        hook_handle = valParam.register_forward_pre_hook(val0_pre_hook)
        hook_handle = valParam.register_forward_hook(val0_forward_hook)
        #------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------

        visOutProjParam = vision_blocks[VisionLayerTrack].self_attn.out_proj
        hook_handle = visOutProjParam.register_forward_pre_hook(visOutProj_pre_hook)
        hook_handle = visOutProjParam.register_forward_hook(visOutProj_forward_hook)

        #------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------

        FC1Param = vision_blocks[VisionLayerTrack].mlp.fc1
        hook_handle = FC1Param.register_forward_pre_hook(FC1_pre_hook)
        hook_handle = FC1Param.register_forward_hook(FC1_forward_hook)
        #------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------

        FC2Param = vision_blocks[VisionLayerTrack].mlp.fc2
        hook_handle = FC2Param.register_forward_pre_hook(FC2_pre_hook)
        hook_handle = FC2Param.register_forward_hook(FC2_forward_hook)

        #------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------

        #------------ multimodal projection hook --------------
        MulModProjParam = get_multi_modal_projector(model)
        hook_handle = MulModProjParam.register_forward_pre_hook(MulModProj_pre_hook)
        hook_handle = MulModProjParam.register_forward_hook(MulModProj_forward_hook)
        # ------- language hooks begin ----------------

        qryLanParam = language_layers[LanLayerTrack].self_attn.q_proj
        hook_handle = qryLanParam.register_forward_pre_hook(qryLan_pre_hook)
        hook_handle = qryLanParam.register_forward_hook(qryLan_forward_hook)


        keyLanParam = language_layers[LanLayerTrack].self_attn.k_proj
        hook_handle = keyLanParam.register_forward_pre_hook(keyLan_pre_hook)
        hook_handle = keyLanParam.register_forward_hook(keyLan_forward_hook)

        valLanParam = language_layers[LanLayerTrack].self_attn.v_proj
        hook_handle = valLanParam.register_forward_pre_hook(valLan_pre_hook)
        hook_handle = valLanParam.register_forward_hook(valLan_forward_hook)


        gate_proj = language_layers[LanLayerTrack].mlp.gate_proj
        hook_handle = gate_proj.register_forward_pre_hook(gate_proj_pre_hook)
        hook_handle = gate_proj.register_forward_hook(gate_proj_forward_hook)

        up_proj = language_layers[LanLayerTrack].mlp.up_proj
        hook_handle = up_proj.register_forward_pre_hook(up_proj_pre_hook)
        hook_handle = up_proj.register_forward_hook(up_proj_forward_hook)


        down_proj = language_layers[LanLayerTrack].mlp.down_proj
        hook_handle = down_proj.register_forward_pre_hook(down_proj_pre_hook)
        hook_handle = down_proj.register_forward_hook(down_proj_forward_hook)

        allTopRightSingularVectors = [getTopRightSingularVectorForAttentioHeads(qryParam),
                                      getTopLeftSingularVectorForAttentioHeads(qryParam),

                                      getTopRightSingularVectorForAttentioHeads(keyParam),
                                      getTopLeftSingularVectorForAttentioHeads(keyParam),

                                      getTopRightSingularVectorForAttentioHeads(valParam),
                                      getTopLeftSingularVectorForAttentioHeads(valParam),

                                      getTopRightSingularVector(visOutProjParam),
                                      getTopLeftSingularVector(visOutProjParam),

                                      getTopRightSingularVector(FC1Param),
                                      getTopLeftSingularVector(FC1Param),

                                      getTopRightSingularVector(FC2Param),
                                      getTopLeftSingularVector(FC2Param),

                                      getTopRightSingularVectorForMMproj(MulModProjParam),
                                      getTopLeftSingularVectorForMMproj(MulModProjParam),

                                      getTopRightSingularVectorForLanAttentioHeads(qryLanParam, num_headsT),
                                      getTopLeftSingularVectorForLanAttentioHeads(qryLanParam, num_headsT),

                                      getTopRightSingularVectorForLanAttentioHeads(keyLanParam, num_kv_headsT),
                                      getTopLeftSingularVectorForLanAttentioHeads(keyLanParam, num_kv_headsT),

                                      getTopRightSingularVectorForLanAttentioHeads(valLanParam, num_kv_headsT),
                                      getTopLeftSingularVectorForLanAttentioHeads(valLanParam, num_kv_headsT),

                                      getTopRightSingularVector(gate_proj),
                                      getTopLeftSingularVector(gate_proj),

                                      getTopRightSingularVector(up_proj),
                                      getTopLeftSingularVector(up_proj),

                                      getTopRightSingularVector(down_proj),
                                      getTopLeftSingularVector(down_proj)]

    # ---------------- hook ----------------------

    point_labels = [
    "query proj", "key proj", "value proj", "att output\nproj",
    "MLP fc1\n(vis)", "MLP fc2\n(vis)", "Vis-to-lan\nproj", "query proj\n",
    "key proj\n", "value proj\n", "MLP gate\nproj", "MLP up\nproj",
    "MLP down\nproj"
    ]

    if attackMode == "vis":
        # 6 vision points (query, key, value, att output proj, fc1, fc2) --
        # one fewer than Qwen2.5-VL's 7, since Gemma3's vision MLP has only
        # two projections (fc1/fc2) instead of three (gate/up/down).
        point_labels = point_labels[:6]
    else:
        # Language items start at index 7 (6 vision points + 1 merger point).
        point_labels = point_labels[7:]

    PostAttackAlignments = []
    PreAttackAlignments = []
    AlignmnetIncreases = []
    NumSamplesConsidered = 50
    AggregationOverFlattenedAlignmentDistributionsOriginal = []
    AggregationFlattenedAlignmentDistributionsAdversary = []
    for attackSample in range(1, NumSamplesConsidered):
        # Same dataset directory / filename convention used by the existing
        # Gemma3 examiner and BSA trainer, so we load the exact same
        # original images the perturbations were trained against.
        IMAGE_PATH = f"../interpretAttacks/gemma_attack/dataSamplesForQuant/{attackSample}.JPEG"

        pil = Image.open(IMAGE_PATH).convert("RGB")
        x_orig01 = pil_to_tensor01(pil).to(device)

        # Build template inputs ONCE (inserts image tokens in input_ids)
        template_inputs = build_template_inputs(processor, QUESTION, pil, device)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # BSA adversarial-noise loading convention: ported VERBATIM from
        # gemma_attack/RealSpectralSubSpaceAlignmentPostAttackExaminerFullDistancesOverlap.py,
        # since Gemma3's BSA trainer names its output .pt files with these
        # towardsNull / lanMLP / visMLP / balancingAlpha tags (no Qwen
        # equivalent exists -- Qwen's own BSA trainer uses a much simpler
        # filename with no such tags).
        towardsNullR = 0.15
        AttackStartLayerR = 0
        numLayerstAtAtimeR = 2
        whichMLPR = "up_proj"
        whichMLPvisR = "fc2"
        balancingAlphaR = 0.5

        adv_noise_path = (
            f"../interpretAttacks/gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
            f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
            f"AttackStartLayer_{AttackStartLayerR}_numLayerstAtAtime_{numLayerstAtAtimeR}_"
            f"num_steps_{num_steps}_towardsNull_{towardsNullR}_"
            f"lanMLP_{whichMLPR}_visMLP_{whichMLPvisR}_"
            f"lanLayers_upto4_visLayers_all_balancingAlpha_{balancingAlphaR}.pt"
        )

        best_delta = torch.load(adv_noise_path, map_location=device).to(device=device, dtype=x_orig01.dtype)

        # Build a "weak adversary" - gaussian noise scaled to the same
        # L_inf epsilon bound as the trained ("strong") adversary. The
        # clamping inside adam_attack_original_space (x_orig01 +/- epsilon)
        # applies to whatever delta is passed in, so this receives the
        # exact same L_inf bound treatment as best_delta.
        weak_delta = torch.randn_like(best_delta) * epsilon

        x_adv01, best_pert, RightSingularInputAlignmentAgainstAdversary, FlattenedAlignmentDistributionsAdversary = adam_attack_original_space(
            model=model,
            processor=processor,
            template_inputs=template_inputs,
            x_orig01=x_orig01,
            attck_type=attck_type,
            num_steps=num_steps,
            lr=lr,
            epsilon=epsilon,
            device=device,
            AttackStartLayer = AttackStartLayer,
            numLayerstAtAtime = numLayerstAtAtime,
            allTopRightSingularVectors = allTopRightSingularVectors,
            best_delta = best_delta
        )
        if attackMode == "vis":
            RightSingularInputAlignmentAgainstAdversary = RightSingularInputAlignmentAgainstAdversary[:6]
            FlattenedAlignmentDistributionsAdversary = FlattenedAlignmentDistributionsAdversary[:6]
        else:
            RightSingularInputAlignmentAgainstAdversary = RightSingularInputAlignmentAgainstAdversary[7:]
            FlattenedAlignmentDistributionsAdversary = FlattenedAlignmentDistributionsAdversary[7:]

        final = RightSingularInputAlignmentAgainstAdversary
        PostAttackAlignments.append(RightSingularInputAlignmentAgainstAdversary)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------
        x_adv01, best_pert, RightSingularInputAlignmentAgainstOriginal, FlattenedAlignmentDistributionsOriginal = adam_attack_original_space(
            model=model,
            processor=processor,
            template_inputs=template_inputs,
            x_orig01=x_orig01,
            attck_type=attck_type,
            num_steps=num_steps,
            lr=lr,
            epsilon=epsilon,
            device=device,
            AttackStartLayer = AttackStartLayer,
            numLayerstAtAtime = numLayerstAtAtime,
            allTopRightSingularVectors = allTopRightSingularVectors,
            best_delta = best_delta*0
        )

        if attackMode == "vis":
            RightSingularInputAlignmentAgainstOriginal = RightSingularInputAlignmentAgainstOriginal[:6]
            FlattenedAlignmentDistributionsOriginal = FlattenedAlignmentDistributionsOriginal[:6]
        else:
            RightSingularInputAlignmentAgainstOriginal = RightSingularInputAlignmentAgainstOriginal[7:]
            FlattenedAlignmentDistributionsOriginal = FlattenedAlignmentDistributionsOriginal[7:]

        # Third pass - weak (gaussian, same L_inf bound) adversary.
        x_adv01, best_pert, RightSingularInputAlignmentAgainstWeak, FlattenedAlignmentDistributionsWeak = adam_attack_original_space(
            model=model,
            processor=processor,
            template_inputs=template_inputs,
            x_orig01=x_orig01,
            attck_type=attck_type,
            num_steps=num_steps,
            lr=lr,
            epsilon=epsilon,
            device=device,
            AttackStartLayer = AttackStartLayer,
            numLayerstAtAtime = numLayerstAtAtime,
            allTopRightSingularVectors = allTopRightSingularVectors,
            best_delta = weak_delta
        )

        if attackMode == "vis":
            RightSingularInputAlignmentAgainstWeak = RightSingularInputAlignmentAgainstWeak[:6]
            FlattenedAlignmentDistributionsWeak = FlattenedAlignmentDistributionsWeak[:6]
        else:
            RightSingularInputAlignmentAgainstWeak = RightSingularInputAlignmentAgainstWeak[7:]
            FlattenedAlignmentDistributionsWeak = FlattenedAlignmentDistributionsWeak[7:]

        # Compute the L2 difference between the ORIGINAL coeffs and each
        # adversary's coeffs BEFORE doing the token-wise averaging (i.e. at
        # the same stage where the code used to do "coeffs.mean(dim=1)").
        # We square the elementwise difference, apply the exact same
        # ".mean(dim=1)" reduction the helper functions used to apply to
        # "coeffs" itself, then sqrt to get an (RMS-style) L2 distance.
        DiffStrongThisSample = []
        DiffWeakThisSample = []
        for i in range(len(FlattenedAlignmentDistributionsOriginal)):
            orig_c = FlattenedAlignmentDistributionsOriginal[i]
            adv_c = FlattenedAlignmentDistributionsAdversary[i]
            weak_c = FlattenedAlignmentDistributionsWeak[i]

            # MODIFIED vs the original Gemma examiner (robustness fix ported
            # from qwen/Qwen2p5FullDistancesOverlap.py): reduce over the
            # TOKEN axis explicitly instead of a hard-coded dim=1. Gemma3's
            # image processor uses a fixed target resolution, so N is
            # constant across samples here (unlike Qwen2.5-VL's dynamic
            # resolution) and dim=1 alone would have "worked" for the
            # attention-head-style coeffs -- but for the plain-Linear coeffs
            # (shape N, D_basis) dim=1 actually reduces over D_basis, not
            # tokens. Detecting the true token axis makes this correct
            # regardless of tensor shape.
            token_dim = 0 if orig_c.dim() == 2 else 1

            diffStrong = (orig_c - adv_c).pow(2).mean(dim=token_dim).sqrt().flatten()
            diffWeak = (orig_c - weak_c).pow(2).mean(dim=token_dim).sqrt().flatten()

            DiffStrongThisSample.append(diffStrong)
            DiffWeakThisSample.append(diffWeak)

        AggregationOverFlattenedAlignmentDistributionsOriginal.append(DiffWeakThisSample)
        AggregationFlattenedAlignmentDistributionsAdversary.append(DiffStrongThisSample)

        averagedAggregationOverFlattenedAlignmentDistributionsOriginal = [
            torch.stack(elements).mean(dim=0)
            for elements in zip(*AggregationOverFlattenedAlignmentDistributionsOriginal)
        ]

        averagedAggregationFlattenedAlignmentDistributionsAdversary = [
            torch.stack(elements).mean(dim=0)
            for elements in zip(*AggregationFlattenedAlignmentDistributionsAdversary)
        ]

        # MODIFIED vs the original Gemma examiner (bug fix ported from
        # qwen/Qwen2p5FullDistancesOverlap.py): these must use .std(dim=0),
        # not another .mean(dim=0) -- the original examiner computed the
        # SAME mean twice under a "STD" variable name.
        averagedAggregationOverFlattenedAlignmentDistributionsOriginalSTD = [
            torch.stack(elements).std(dim=0)
            for elements in zip(*AggregationOverFlattenedAlignmentDistributionsOriginal)
        ]

        averagedAggregationFlattenedAlignmentDistributionsAdversarySTD = [
            torch.stack(elements).std(dim=0)
            for elements in zip(*AggregationFlattenedAlignmentDistributionsAdversary)
        ]

    for i in range(len(averagedAggregationOverFlattenedAlignmentDistributionsOriginal)):
            print("FlattenedAlignmentDistributionsAdversary[i].shape", averagedAggregationFlattenedAlignmentDistributionsAdversary[i].shape)
            print("FlattenedAlignmentDistributionsOriginal[i].shape", averagedAggregationOverFlattenedAlignmentDistributionsOriginal[i].shape)

            weak = averagedAggregationOverFlattenedAlignmentDistributionsOriginal[i].detach().to(torch.float32).cpu().numpy()
            strong = averagedAggregationFlattenedAlignmentDistributionsAdversary[i].detach().to(torch.float32).cpu().numpy()

            weak = (weak)
            strong = (strong)

            weak_mean = np.mean(weak)
            strong_mean = np.mean(strong)

            weak_norm = weak / weak_mean
            strong_norm = strong / strong_mean

            L = len(weak)
            x = np.arange(L)

            label = point_labels[i].replace("\n", " ")

            fig, ax = plt.subplots(figsize=(10, 3.5))

            ax.bar(x, strong, width=1.0, color="red", edgecolor="none", alpha=0.6, label="Strong Adversary (Trained) vs Original", zorder=2)
            ax.bar(x, weak, width=1.0, color="blue", edgecolor="none", alpha=0.6, label="Weak Adversary (Gaussian) vs Original", zorder=1)

            ax.set_title(f"{label} — L2 Distance: Weak vs Strong Adversary")
            ax.set_xlabel("<- top singular vector   |   bottom singular vector ->")
            ax.set_ylabel("L2 Distance")
            ax.legend()

            plt.tight_layout()

            save_dir = f"gemma_attack/OverlapDistancesAvg"

            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(
                save_dir,
                f"Bar_{label.replace(' ', '_')}_attackSample_{attackSample}_attackMode_{attackMode}_LanLayerTrack_{LanLayerTrack})_VisionLayerTrack_{VisionLayerTrack}.png"
            )
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()
            print(f"Saved: {save_path}")

            weak_vs_strong_l2 = float(np.linalg.norm(strong - weak))
            csv_path = os.path.join(save_dir, "weak_vs_strong_l2_summary.csv")
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as csv_f:
                csv_writer = csv.writer(csv_f)
                if write_header:
                    csv_writer.writerow([
                        "LanLayerTrack", "VisionLayerTrack", "attackMode",
                        "point_label", "weak_vs_strong_l2_distance"
                    ])
                csv_writer.writerow([
                    LanLayerTrack, VisionLayerTrack, attackMode,
                    label, weak_vs_strong_l2
                ])
            print(f"Logged weak-vs-strong L2 distance ({weak_vs_strong_l2:.6f}) to: {csv_path}")


            #--------------------------------------------------------------------            #--------------------------------------------------------------------            #--------------------------------------------------------------------

            weak_std = averagedAggregationOverFlattenedAlignmentDistributionsOriginalSTD[i].detach().to(torch.float32).cpu().numpy()
            strong_std = averagedAggregationFlattenedAlignmentDistributionsAdversarySTD[i].detach().to(torch.float32).cpu().numpy()

            weak_std_norm = weak_std / weak_mean
            strong_std_norm = strong_std / strong_mean

            fig, ax = plt.subplots(figsize=(10, 3.5))

            ax.fill_between(x, weak - weak_std, weak + weak_std, color="blue", alpha=0.35, linewidth=0, zorder=2.1)
            ax.plot(x, weak, color="blue", linewidth=0.6, zorder=2.2)

            ax.fill_between(x, strong - strong_std, strong + strong_std, color="red", alpha=0.35, linewidth=0, zorder=2.3)
            ax.plot(x, strong, color="red", linewidth=0.6, zorder=2.4)

            ax.set_title(f"{label} — L2 Distance: Weak vs Strong Adversary (Mean ± STD)")
            ax.set_xlabel("<- top singular vector   |   bottom singular vector ->")
            ax.set_ylabel("L2 Distance")
            ax.legend()

            plt.tight_layout()

            save_dirBands = f"gemma_attack/OverlapDistancesAvgStdBands"
            os.makedirs(save_dirBands, exist_ok=True)
            save_pathBands = os.path.join(
                save_dirBands,
                f"Bar_{label.replace(' ', '_')}_attackSample_{attackSample}_attackMode_{attackMode}_LanLayerTrack_{LanLayerTrack})_VisionLayerTrack_{VisionLayerTrack}.png"
            )
            plt.savefig(save_pathBands, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {save_pathBands}")

            #--------------------------------------------------------------------            #--------------------------------------------------------------------            #--------------------------------------------------------------------

            fig, ax = plt.subplots(figsize=(10, 3.5))

            ax.fill_between(x, weak_norm - weak_std_norm, weak_norm + weak_std_norm, color="blue", alpha=0.2, linewidth=0, zorder=2.1)
            ax.plot(x, weak_norm, color="blue", linewidth=0.6, zorder=2.2)

            ax.fill_between(x, strong_norm - strong_std_norm, strong_norm + strong_std_norm, color="red", alpha=0.2, linewidth=0, zorder=2.3)
            ax.plot(x, strong_norm, color="red", linewidth=0.6, zorder=2.4)

            ax.set_title(f"{label} — L2 Distance: Weak vs Strong Adversary (Mean ± STD)")
            ax.set_xlabel("<- top singular vector   |   bottom singular vector ->")
            ax.set_ylabel("L2 Distance")
            ax.legend()

            plt.tight_layout()

            save_dirBandsNorm = f"gemma_attack/OverlapDistancesAvgStdBandsNormalized"
            os.makedirs(save_dirBandsNorm, exist_ok=True)
            save_pathBandsNorm = os.path.join(
                save_dirBandsNorm,
                f"Bar_{label.replace(' ', '_')}_attackSample_{attackSample}_attackMode_{attackMode}_LanLayerTrack_{LanLayerTrack})_VisionLayerTrack_{VisionLayerTrack}.png"
            )
            plt.savefig(save_pathBandsNorm, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {save_pathBandsNorm}")



if __name__ == "__main__":
    main()
