

'''


---------------------------------------------------------------------------

export CUDA_VISIBLE_DEVICES=4
conda deactivate
cd spectralShift/
conda activate llava15
export PYTHONNOUSERSITE=1
python llava_attack/llava1p5DetectEachAdversary.py --attck_type bsa --desired_norm_l_inf 0.002 --thickEpsilon 0.06 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --detectionThreshold 0.95
python llava_attack/llava1p5DetectEachAdversary.py --attck_type bsa --desired_norm_l_inf 0.003 --thickEpsilon 0.06 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --detectionThreshold 0.95
python llava_attack/llava1p5DetectEachAdversary.py --attck_type bsa --desired_norm_l_inf 0.004 --thickEpsilon 0.06 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --detectionThreshold 0.95
python llava_attack/llava1p5DetectEachAdversary.py --attck_type bsa --desired_norm_l_inf 0.005 --thickEpsilon 0.06 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --detectionThreshold 0.95



export CUDA_VISIBLE_DEVICES=3
conda deactivate
cd spectralShift/
conda activate llava15
export PYTHONNOUSERSITE=1
python llava_attack/llava1p5DetectEachAdversary.py --attck_type nllm --desired_norm_l_inf 0.002 --thickEpsilon 0.06 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --detectionThreshold 0.95
python llava_attack/llava1p5DetectEachAdversary.py --attck_type nllm --desired_norm_l_inf 0.003 --thickEpsilon 0.06 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --detectionThreshold 0.95
python llava_attack/llava1p5DetectEachAdversary.py --attck_type nllm --desired_norm_l_inf 0.004 --thickEpsilon 0.06 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --detectionThreshold 0.95
python llava_attack/llava1p5DetectEachAdversary.py --attck_type nllm --desired_norm_l_inf 0.005 --thickEpsilon 0.06 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --detectionThreshold 0.95


export CUDA_VISIBLE_DEVICES=2
conda deactivate
cd spectralShift/
conda activate llava15
export PYTHONNOUSERSITE=1
python llava_attack/llava1p5DetectEachAdversary.py --attck_type ega --desired_norm_l_inf 0.002 --thickEpsilon 0.06 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --detectionThreshold 0.95
python llava_attack/llava1p5DetectEachAdversary.py --attck_type ega --desired_norm_l_inf 0.003 --thickEpsilon 0.06 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --detectionThreshold 0.95
python llava_attack/llava1p5DetectEachAdversary.py --attck_type ega --desired_norm_l_inf 0.004 --thickEpsilon 0.06 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --detectionThreshold 0.95
python llava_attack/llava1p5DetectEachAdversary.py --attck_type ega --desired_norm_l_inf 0.005 --thickEpsilon 0.06 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --detectionThreshold 0.95





export CUDA_VISIBLE_DEVICES=1
conda deactivate
cd spectralShift/
conda activate llava15
export PYTHONNOUSERSITE=1
python llava_attack/llava1p5DetectEachAdversary.py --attck_type justNoise --desired_norm_l_inf 0.005 --thickEpsilon 0.06 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --detectionThreshold 0.95 
python llava_attack/llava1p5DetectEachAdversary.py --attck_type justNoise --desired_norm_l_inf 0.004 --thickEpsilon 0.06 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --detectionThreshold 0.95
python llava_attack/llava1p5DetectEachAdversary.py --attck_type justNoise --desired_norm_l_inf 0.003 --thickEpsilon 0.06 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --detectionThreshold 0.95
python llava_attack/llava1p5DetectEachAdversary.py --attck_type justNoise --desired_norm_l_inf 0.002 --thickEpsilon 0.06 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --detectionThreshold 0.95



export CUDA_VISIBLE_DEVICES=3
conda deactivate
cd spectralShift/
conda activate llava15
export PYTHONNOUSERSITE=1
python llava_attack/llava1p5DetectEachAdversary.py --attck_type cleanImages --thickEpsilon 0.06 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --detectionThreshold 0.95 






export CUDA_VISIBLE_DEVICES=3
conda deactivate
cd spectralShift/
conda activate llava15
export PYTHONNOUSERSITE=1
python llava_attack/llava1p5DetectEachAdversary.py --attck_type bsa --desired_norm_l_inf 0.006 --thickEpsilon 0.06 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --detectionThreshold 0.95
python llava_attack/llava1p5DetectEachAdversary.py --attck_type nllm --desired_norm_l_inf 0.006 --thickEpsilon 0.06 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --detectionThreshold 0.95
python llava_attack/llava1p5DetectEachAdversary.py --attck_type nllm --desired_norm_l_inf 0.007 --thickEpsilon 0.06 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --detectionThreshold 0.95
python llava_attack/llava1p5DetectEachAdversary.py --attck_type ega --desired_norm_l_inf 0.006 --thickEpsilon 0.06 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --detectionThreshold 0.95
python llava_attack/llava1p5DetectEachAdversary.py --attck_type justNoise --desired_norm_l_inf 0.006 --thickEpsilon 0.06 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --detectionThreshold 0.95 
python llava_attack/llava1p5DetectEachAdversary.py --attck_type justNoise --desired_norm_l_inf 0.006 --thickEpsilon 0.06 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --detectionThreshold 0.95 



 (local-variation table from the Qwen runs removed -- not valid for LLaVA)



'''




import os
# MODIFIED vs Gemma (adopted from qwen/QwenUntargeted_BSA.py and
# qwen/QwenUntargeted_BSA_inference.py): this must be set before any CUDA/cuBLAS
# context is created for deterministic matmuls to actually take effect, so it
# has to come immediately after `import os`, before torch is imported.
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
from transformers import (
    LlavaForConditionalGeneration,
    CLIPImageProcessor,
    LlamaTokenizer,
)
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
# Differentiable preprocessing (LLaVA-1.5 / CLIP)
# MODIFIED vs Qwen: LLaVA's CLIP image processor is a fixed-size
# resize(shortest edge) + center-crop (336x336) + normalize, and returns a
# plain (1,3,336,336) pixel_values tensor with no image_grid_thw. Ported
# verbatim from llava_attack/Llava1p5FullDistancesOverlap.py so the same
# perturbations line up with the pipeline they were trained with.
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
# Build template inputs ONCE (IMPORTANT)
# MODIFIED vs Qwen: LLaVA-1.5 uses the fixed "USER: <image>\n...\nASSISTANT:"
# prompt tokenized with the raw LlamaTokenizer (no chat template, no
# LlavaProcessor), same as llava_attack/Llava1p5FullDistancesOverlap.py.
# ----------------------------
def build_template_inputs(tokenizer, question: str, device):
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    enc = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


# ----------------------------
# Generation helper (uses template, swaps pixel_values)
# MODIFIED vs Qwen: no image_grid_thw for LLaVA; decode with the raw tokenizer.
# ----------------------------
def run_generation_with_pixel_values(model, tokenizer, template_inputs, pixel_values, max_new_tokens=128):
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
    return tokenizer.decode(gen_only[0], skip_special_tokens=True)


def _drop_batch_dim_if_present(t):
    """
    ADDED vs Gemma (necessary correctness fix): Gemma's SigLIP vision tower
    and its language model both keep a leading batch dimension of size 1, so
    the original code's `InputToLayer[0]` reliably drops that batch dim and
    leaves a (N, d_model) tensor. Qwen2.5-VL's vision transformer instead
    runs on a single FLAT (total_tokens, hidden_dim) tensor with NO batch
    dimension at all (patches from the image are concatenated along dim 0
    directly). Blindly doing `InputToLayer[0]` on that tensor would select
    just the first patch/token instead of dropping a batch dim, and crashes
    downstream (F.normalize on a 1-D tensor with dim=1). This helper detects
    which convention we're in and handles both, so the same alignment
    functions work unchanged for vision (unbatched) and language (batched)
    hooks.
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

    #print("H_hat.shape", H_hat.shape)
    #print("V_hat.shape", V_hat.shape)
    coeffs = H_hat @ V_hat.T        # (N, k)

    #print("coeffs.shape", coeffs.shape)

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
# MODIFIED vs Qwen: robust getters for LLaVA-1.5's CLIP vision encoder layers /
# LLaMA language layers / multi_modal_projector. Old transformers (matching
# llava_attack/model_parameters.txt) expose model.vision_tower,
# model.multi_modal_projector and model.language_model.model.layers; newer
# versions wrap them in model.model.*, so we probe for both.
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
def get_vision_module_and_blocks(model):
    for root in (model, getattr(model, "model", None)):
        if root is None:
            continue
        vt = getattr(root, "vision_tower", None)
        if vt is not None and hasattr(vt, "vision_model"):
            return vt, vt.vision_model.encoder.layers
    raise RuntimeError("Could not find LLaVA vision tower encoder layers.")


def get_multi_modal_projector(model):
    for root in (model, getattr(model, "model", None)):
        if root is not None and hasattr(root, "multi_modal_projector"):
            return root.multi_modal_projector
    raise RuntimeError("Could not find LLaVA multi_modal_projector.")


def get_language_module_and_layers(model):
    lm = getattr(model, "language_model", None)
    if lm is not None:
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model, lm.model.layers
        if hasattr(lm, "layers"):
            return lm, lm.layers
    inner = getattr(model, "model", None)
    if inner is not None:
        lm = getattr(inner, "language_model", None)
        if lm is not None and hasattr(lm, "layers"):
            return lm, lm.layers
        if hasattr(inner, "layers"):
            return inner, inner.layers
    raise RuntimeError("Could not find LLaVA language-model layers.")


def getImageLocalVar(x_ad):
    dx = x_ad[:, :, :, 1:] - x_ad[:, :, :, :-1]
    dy = x_ad[:, :, 1:, :] - x_ad[:, :, :-1, :]
    ObLocVar = (dx.abs().mean() + dy.abs().mean()) / 2
    return ObLocVar


'''def getStableobserved_local_var(x_orig01, epsilon, times):
    collectobserved_local_var = []
    for i in range(times):

        checkItthatWay = torch.randn_like(x_orig01)
        checkItthatWayNormal = 2 * (checkItthatWay - checkItthatWay.min()) / (checkItthatWay.max() - checkItthatWay.min()) - 1

        justNoiseHere = checkItthatWayNormal * epsilon

        x_adv01 = (x_orig01 + justNoiseHere).clamp(0.0, 1.0)
        x_adv01 = torch.max(torch.min(x_adv01, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

        observed_local_var = getImageLocalVar(x_adv01)

        collectobserved_local_var.append(observed_local_var.item())

    collectobserved_local_varArr = np.array(collectobserved_local_var)
    collectobserved_local_varArrMean = np.mean(collectobserved_local_varArr)
    return collectobserved_local_varArrMean'''


def getImageLocalVarListPerPerturbation(x_orig01, Start, End, NumPts):
    obLocVarList = []
    AllEpsilon = np.linspace(Start, End, NumPts)
    #print("AllEpsilon", AllEpsilon)
    for eps in AllEpsilon:
        checkItthatWay = torch.randn_like(x_orig01)
        checkItthatWayNormal = 2 * (checkItthatWay - checkItthatWay.min()) / (checkItthatWay.max() - checkItthatWay.min()) - 1
        justNoiseHere = checkItthatWayNormal * eps
        x_adv01 = (x_orig01 + justNoiseHere).clamp(0.0, 1.0)
        x_adv01 = torch.max(torch.min(x_adv01, x_orig01 + eps), x_orig01 - eps).clamp(0.0, 1.0)
        observed_local_var = getImageLocalVar(x_adv01)
        #observed_local_var = getStableobserved_local_var(x_orig01, eps, 1)
        #print("observed_local_var", observed_local_var.item())
        obLocVarList.append(observed_local_var.item())
    obLocVarListArr = np.array(obLocVarList)
    return obLocVarListArr, AllEpsilon




def _resolve_vision_config(model):
    cfg = model.config
    return getattr(cfg, "vision_config", cfg)


def _resolve_text_config(model):
    cfg = model.config
    return getattr(cfg, "text_config", cfg)


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

# MODIFIED vs Qwen: LLaVA's CLIP vision MLP is a plain 2-layer GELU FFN
# (mlp.fc1 / mlp.fc2), not Qwen's 3-projection SwiGLU (gate/up/down), so the
# three vision hooks visGate / visUp / visDown are replaced by visFC1 / visFC2.
visFC1_inputs = {}
def visFC1_pre_hook(module, inputs):
    visFC1_inputs["visFC1_in"] = inputs[0]
visFC1_outputs = {}
def visFC1_forward_hook(module, inputs, output):
    visFC1_outputs["visFC1_out"] = output
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
visFC2_inputs = {}
def visFC2_pre_hook(module, inputs):
    visFC2_inputs["visFC2_in"] = inputs[0]
visFC2_outputs = {}
def visFC2_forward_hook(module, inputs, output):
    visFC2_outputs["visFC2_out"] = output
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
    image_processor,
    template_inputs,
    x_orig01,               # (1,3,H0,W0) in [0,1]
    attck_type: str,
    whatKindOfAdversary: str,
    num_steps: int,
    lr: float,
    epsilon: float,         # L_inf bound in ORIGINAL pixel space [0,1]
    thickEpsilon: float,
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
        pv_clean_fixed = llava_preprocess_differentiable(x_orig01, image_processor)

        clean_inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
        clean_inputs["pixel_values"] = pv_clean_fixed
        clean_inputs["labels"] = template_inputs["input_ids"]
        clean_inputs["use_cache"] = False
        outputsN = model(**clean_inputs, output_hidden_states=True, return_dict=True)
        hiddStateLen = len(outputsN.hidden_states)
        #print(" Number of hidden states is: ", hiddStateLen)

        startPos = AttackStartLayer
        endPos = startPos + numLayerstAtAtime
        #print("endPos", endPos)
        #print("startPos", startPos)
        if endPos > hiddStateLen:
            raise ValueError(
                f"endPos ({endPos}) exceeds number of hidden states ({hiddStateLen})"
            )

        #----------------------attention head hyper parameters extraction------------------------
        # MODIFIED vs Qwen: CLIP vision config names the head count
        # num_attention_heads (16); LLaVA's LLaMA-7B has no GQA
        # (num_key_value_heads == num_attention_heads == 32).
        vision_module, vision_blocks = get_vision_module_and_blocks(model)
        vision_cfg = _resolve_vision_config(model)
        text_cfg = _resolve_text_config(model)

        num_heads = getattr(vision_cfg, "num_attention_heads", getattr(vision_cfg, "num_heads", 16))
        d_model = getattr(vision_cfg, "hidden_size", vision_blocks[0].self_attn.q_proj.weight.shape[1])
        d_head = d_model // num_heads
        #----------------------attention head hyper parameters extraction------------------------

        d_modelT = getattr(text_cfg, "hidden_size")
        num_headsT = getattr(text_cfg, "num_attention_heads")
        num_kv_headsT = getattr(text_cfg, "num_key_value_heads", num_headsT)


    adv_inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
    adv_inputs["labels"] = template_inputs["input_ids"]
    adv_inputs["use_cache"] = False
    RightSingularInputAlignmentWhole = []
    LeftSingularOutputStepAlignmentWhole = []
    #for step in range(1):




    if attck_type == "justNoise":
        checkItthatWay = torch.randn_like(x_orig01)
        checkItthatWayNormal = 2 * (checkItthatWay - checkItthatWay.min()) / (checkItthatWay.max() - checkItthatWay.min()) - 1

        justNoiseHere = checkItthatWayNormal * epsilon

        x_adv01 = (x_orig01 + justNoiseHere).clamp(0.0, 1.0)
        x_adv01 = torch.max(torch.min(x_adv01, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

        x_adv01_created = x_adv01
        #linf_norm = torch.norm((x_adv01_created-x_orig01), p=float('inf'))
        #print("linf_norm", linf_norm)

    elif attck_type == "cleanImages":   # MODIFIED vs Qwen: was `if`, which let justNoise fall into the `else` below
        #checkItthatWay = torch.randn_like(x_orig01)
        #checkItthatWayNormal = 2 * (checkItthatWay - checkItthatWay.min()) / (checkItthatWay.max() - checkItthatWay.min()) - 1

        #justNoiseHere = checkItthatWayNormal * epsilon

        #x_adv01 = (x_orig01 + justNoiseHere).clamp(0.0, 1.0)
        #x_adv01 = torch.max(torch.min(x_adv01, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)
        x_adv01 = x_orig01
        x_adv01_created = x_adv01
        #linf_norm = torch.norm((x_adv01_created-x_orig01), p=float('inf'))
        #print("linf_norm", linf_norm)

    else:
        x_adv01 = (x_orig01 + delta).clamp(0.0, 1.0)
        x_adv01 = torch.max(torch.min(x_adv01, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)
        x_adv01_created = x_adv01

    if whatKindOfAdversary == "weakened":

        #x_adv01 = (x_orig01 + delta).clamp(0.0, 1.0)
        #x_adv01 = torch.max(torch.min(x_adv01, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

        checkItthatWay = torch.randn_like(x_adv01)
        checkItthatWayNormal = 2 * (checkItthatWay - checkItthatWay.min()) / (checkItthatWay.max() - checkItthatWay.min()) - 1

        #weak_delta = torch.randn_like(best_delta) * thickEpsilon

        weak_delta = checkItthatWayNormal * thickEpsilon

        x_adv01 = (x_adv01_created + weak_delta).clamp(0.0, 1.0)
        x_adv01 = torch.max(torch.min(x_adv01, x_adv01_created + thickEpsilon), x_adv01_created - thickEpsilon).clamp(0.0, 1.0)
        


    # preprocess adv (must be differentiable)
    pv_adv = llava_preprocess_differentiable(x_adv01, image_processor)

    adv_inputs["pixel_values"] = pv_adv

    outputs = model(**adv_inputs, output_hidden_states=True, return_dict=True)


    #if step%20==0:
    RightSingularInputAlignment = []
    LeftSingularOutputStepAlignment = []
    AlignmentDistributions = []
    with torch.no_grad():

        InputToLayer = qry0_inputs.get("qry0_in")
        qry0_in_MAE, ProjDistrib = getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, allTopRightSingularVectors[0])
        #print("qry0_in_MAE", qry0_in_MAE)
        #OutputOfLayer = qry0_outputs.get("qry0_out")
        #qry0_out_AME = getMeanAlignmentWithAttentionHeadTopLeftSingularVector(OutputOfLayer, num_heads, allTopRightSingularVectors[1])
        #print("qry0_in  qry0_out", qry0_in_MAE, qry0_out_AME)
        RightSingularInputAlignment.append(qry0_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        InputToLayer = key0_inputs.get("key0_in")
        key0_in_MAE, ProjDistrib = getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, allTopRightSingularVectors[2])
        #OutputOfLayer = key0_outputs.get("key0_out")
        #key0_out_MAE = getMeanAlignmentWithAttentionHeadTopLeftSingularVector(OutputOfLayer, num_heads, allTopRightSingularVectors[3])
        #print("key0_in_MAE, key0_out_MAE", key0_in_MAE, key0_out_MAE)
        RightSingularInputAlignment.append(key0_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        InputToLayer = val0_inputs.get("val0_in")
        val0_in_MAE, ProjDistrib = getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, allTopRightSingularVectors[4])
        #OutputOfLayer = val0_outputs.get("val0_out")
        #val0_out_MAE = getMeanAlignmentWithAttentionHeadTopLeftSingularVector(OutputOfLayer, num_heads, allTopRightSingularVectors[5])
        #print("val0_in_MAE, val0_out_MAE", val0_in_MAE, val0_out_MAE)
        RightSingularInputAlignment.append(val0_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        InputToLayer = visOutProj_inputs.get("visOutProj_in")
        visOutProj_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[6])
        #OutputOfLayer = visOutProj_outputs.get("visOutProj_out")
        #visOutProj_out_MAE = getMeanAlignmentWithTopLeftSingularVector(OutputOfLayer, allTopRightSingularVectors[7])
        #print("visOutProj_in_MAE, visOutProj_out_MAE", visOutProj_in_MAE ,visOutProj_out_MAE)
        RightSingularInputAlignment.append(visOutProj_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # MODIFIED vs Qwen: visGate/visUp/visDown -> visFC1/visFC2 (CLIP 2-layer MLP)
        InputToLayer = visFC1_inputs.get("visFC1_in")
        visFC1_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[8])
        RightSingularInputAlignment.append(visFC1_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        InputToLayer = visFC2_inputs.get("visFC2_in")
        visFC2_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[10])
        RightSingularInputAlignment.append(visFC2_in_MAE)
        AlignmentDistributions.append(ProjDistrib)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        InputToLayer = MulModProj_inputs.get("MulModProj_in")
        MulModProj_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[12])
        #OutputOfLayer = MulModProj_outputs.get("MulModProj_out")
        #MulModProj_out_MAE = getMeanAlignmentWithTopLeftSingularVector(OutputOfLayer, allTopRightSingularVectors[15])
        #print("MulModProj_in_MAE MulModProj_out_MAE", MulModProj_in_MAE, MulModProj_out_MAE)
        RightSingularInputAlignment.append(MulModProj_in_MAE)
        AlignmentDistributions.append(ProjDistrib)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        InputToLayer = qryLan_inputs.get("qryLan_in")
        qryLan_in_MAE, ProjDistrib = getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, allTopRightSingularVectors[14])
        #OutputOfLayer = qryLan_outputs.get("qryLan_out")
        #qryLan_out_MAE = getMeanAlignmentWithAttentionHeadTopLeftSingularVector(OutputOfLayer, num_headsT, allTopRightSingularVectors[17])
        #print("qryLan_in_MAE qryLan_out_MAE", qryLan_in_MAE, qryLan_out_MAE)
        RightSingularInputAlignment.append(qryLan_in_MAE)
        AlignmentDistributions.append(ProjDistrib)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        InputToLayer = keyLan_inputs.get("keyLan_in")
        keyLan_in_MAE, ProjDistrib = getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, allTopRightSingularVectors[16])
        #OutputOfLayer = keyLan_outputs.get("keyLan_out")
        #keyLan_out_MAE = getMeanAlignmentWithAttentionHeadTopLeftSingularVector(OutputOfLayer, num_kv_headsT, allTopRightSingularVectors[19])
        #print("keyLan_in_MAE keyLan_out_MAE", keyLan_in_MAE, keyLan_out_MAE)
        RightSingularInputAlignment.append(keyLan_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        InputToLayer = valLan_inputs.get("valLan_in")
        valLan_in_MAE, ProjDistrib = getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, allTopRightSingularVectors[18])
        #OutputOfLayer = valLan_outputs.get("valLan_out")
        #valLan_out_MAE = getMeanAlignmentWithAttentionHeadTopLeftSingularVector(OutputOfLayer, num_kv_headsT, allTopRightSingularVectors[21])
        #print(" valLan_in_MAE,  valLan_out_MAE ", valLan_in_MAE,  valLan_out_MAE)
        RightSingularInputAlignment.append(valLan_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        InputToLayer = GateLayerInputs.get("gate_in")
        gate_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[20])
        #OutputOfLayer = GateLayerOutputs.get("gate_out")
        #gate_out_MAE = getMeanAlignmentWithTopLeftSingularVector(OutputOfLayer, allTopRightSingularVectors[23])
        #print("gate_in_MAE, gate_out_MAE", gate_in_MAE, gate_out_MAE)
        RightSingularInputAlignment.append(gate_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        InputToLayer = UpLayerInputs.get("up_in")
        up_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[22])
        #OutputOfLayer = UpLayerOutputs.get("up_out")
        #up_out_MAE = getMeanAlignmentWithTopLeftSingularVector(OutputOfLayer, allTopRightSingularVectors[25])
        #print("up_out, up_out_MAE", up_in_MAE, up_out_MAE)
        RightSingularInputAlignment.append(up_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        InputToLayer = DownLayer_inputs.get("down_in")
        down_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[24])
        #OutputOfLayer = DownLayer_outputs.get("down_out")
        #down_out_MAE = getMeanAlignmentWithTopLeftSingularVector(OutputOfLayer, allTopRightSingularVectors[27])
        #print("down_in_MAE down_out_MAE", down_in_MAE, down_out_MAE)
        RightSingularInputAlignment.append(down_in_MAE)
        AlignmentDistributions.append(ProjDistrib)


    RightSingularInputAlignment = np.array(RightSingularInputAlignment)

    #print("len(AlignmentDistributions)", len(AlignmentDistributions))

    FlattenedAlignmentDistributions = []
    for i in range(len(AlignmentDistributions)):
        #print("AlignmentDistributions[i].shape", AlignmentDistributions[i].flatten().shape)
        # MODIFIED: kept un-flattened (raw coeffs, shape (N,k) or (h,n,k)) instead
        # of AlignmentDistributions[i].flatten(). We need the original
        # (pre-reduction) shape at the call site to compute an L2 difference
        # against another adversary's coeffs BEFORE doing the token-wise
        # averaging that used to happen inside the helper functions above.
        FlattenedAlignmentDistributions.append(AlignmentDistributions[i])
    #RightSingularInputAlignmentWhole.append(RightSingularInputAlignment)


    with torch.no_grad():
        x_adv01_final = (x_orig01 + best_delta).clamp(0.0, 1.0)
        x_adv01_final = torch.max(torch.min(x_adv01_final, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

    #print("len(RightSingularInputAlignment)", len(RightSingularInputAlignment))
    #print("RightSingularInputAlignment", RightSingularInputAlignment)
    return x_adv01_final, best_delta, RightSingularInputAlignment, FlattenedAlignmentDistributions


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="LLaVA-1.5 ORIGINAL-image-space adversarial attack (no squeeze)")
    parser.add_argument("--attck_type", type=str, default="bsa",
                        help="bsa | nllm | ega | justNoise | cleanImages")
    parser.add_argument("--desired_norm_l_inf", type=float, default=0.003,
                        help="epsilon L_inf in ORIGINAL pixel space [0..1]. Try 0.01~0.08")
    parser.add_argument("--thickEpsilon", type=float, default=0.03,
                        help="thickEpsilon L_inf in ORIGINAL pixel space [0..1]. Try 0.01~0.08")
    parser.add_argument("--learningRate", type=float, default=1e-3,
                        help="Adam learning rate")
    parser.add_argument("--num_steps", type=int, default=2000,
                        help="Number of Adam steps")
    parser.add_argument("--attackSample", type=int, default=2,
                    help="which sample")
    parser.add_argument("--AttackStartLayer", type=int, default=0,
                        help="From which layer do you start attack")
    parser.add_argument("--numLayerstAtAtime", type=int, default=2,
                        help="Number of layers taken at a time to attack")
    parser.add_argument("--VisionLayerTrack", type=int, default=0,
                        help="whcih vision layer you want to talk")
    parser.add_argument("--LanLayerTrack", type=int, default=0,
                        help="whcih language layer you want to talk")
    parser.add_argument("--kthSingVec", type=int, default=0,
                    help="Amonhg the k singular vectors which one do you wanty")
    parser.add_argument("--attackMode", type=str, default="lan",
                    help="Which layer were attacked vis or lan")
    parser.add_argument("--detectionThreshold", type=float, default=0.98,
                        help="Adam learning rate")
    #parser.add_argument("--ignoreThreshold", type=float, default=0.1,
    #                    help="Adam learning rate")

    args = parser.parse_args()

    attck_type = args.attck_type
    epsilon = float(args.desired_norm_l_inf)
    thickEpsilon = float(args.thickEpsilon)
    lr = float(args.learningRate)
    num_steps = int(args.num_steps)
    attackSample = int(args.attackSample)
    AttackStartLayer = int(args.AttackStartLayer)
    numLayerstAtAtime = int(args.numLayerstAtAtime)

    VisionLayerTrack = int(args.VisionLayerTrack)
    LanLayerTrack = int(args.LanLayerTrack)

    kthSingVec = int(args.kthSingVec)
    attackMode = str(args.attackMode)
    detectionThreshold = float(args.detectionThreshold)
    #ignoreThreshold = float(args.ignoreThreshold)

    MODEL_PATH = "../LLaVA/llava-1.5-7b-hf"
    QUESTION = "What is shown in this image?"
    MAX_NEW_TOKENS = 128


    #attackMode = "lan"
    #attackMode = "vis"


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"device={device}, dtype={dtype}")

    # MODIFIED vs Qwen: model loading from llava_attack/Llava1p5FullDistancesOverlap.py
    # (raw LlamaTokenizer + CLIPImageProcessor, no LlavaProcessor, fp16).
    print("Loading tokenizer + image_processor...")
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

    #----------------------attention head hyper parameters extraction------------------------
    vision_module, vision_blocks = get_vision_module_and_blocks(model)
    language_module, language_layers = get_language_module_and_layers(model)

    vision_cfg = _resolve_vision_config(model)
    text_cfg = _resolve_text_config(model)

    num_heads = getattr(vision_cfg, "num_attention_heads", getattr(vision_cfg, "num_heads", 16))
    d_model = getattr(vision_cfg, "hidden_size", vision_blocks[0].self_attn.q_proj.weight.shape[1])
    d_head = d_model // num_heads
    #----------------------attention head hyper parameters extraction------------------------

    # ADDED vs Qwen: LLaVA's vision-to-language projector (multi_modal_projector)
    mm_projector = get_multi_modal_projector(model)


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

    # MODIFIED vs Qwen: LLaVA's multi_modal_projector.linear_2 is a standard
    # nn.Linear (4096 -> 4096), hooked and SVD'd exactly like Qwen's merger.mlp[2].

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

    # MODIFIED vs Qwen: CLIP's vision q_proj/k_proj/v_proj are SEPARATE
    # nn.Linear modules (1024x1024), not Qwen's fused attn.qkv, so the
    # getTopRight/LeftSingularVectorForVisionQKV slicing helpers are not needed;
    # getTopRight/LeftSingularVectorForAttentioHeads above are used directly.

    d_modelT = getattr(text_cfg, "hidden_size")
    num_headsT = getattr(text_cfg, "num_attention_heads")
    # LLaVA's LLaMA-7B has no GQA (num_key_value_heads == num_attention_heads == 32);
    # kept generic, same as Qwen.
    num_kv_headsT = getattr(text_cfg, "num_key_value_heads", num_headsT)

    #print("num_headsT", num_headsT)
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


    NumCountsSet = 4
    perAttackSampleProbMaxes = []
    perAttackSampleProbMins = []

    NumTimesYouHitTheMarkPerSample = []
    probsAllSamplesAlllayer = []

    for attackSample in range(1,101):
    #for attackSample in range(1,3):
        countAdvChance = 0
        countIgnore = 0
        NumTimesYouHitTheMark = 0
        NumTimesYouHit = 0

        probMax = torch.tensor(0.0, device=device)
        probMin = torch.tensor(1.0, device=device)

        #attackSample = 39 # check
        probsAlllayers = []
        for LanLayerTrack in range(32):
        #for LanLayerTrack in range(4):

            with torch.no_grad():

                # ADDED vs Qwen: keep the hook handles so they can be removed at the end of
                # this LanLayerTrack iteration. In the Qwen script they were never removed, so
                # hooks from earlier layers/samples kept firing and overwrote the captured inputs.
                hook_handles = []

                # MODIFIED vs Qwen: CLIP has separate q_proj / k_proj / v_proj Linears
                # (Qwen had one fused attn.qkv hooked three times).
                qryParam = vision_blocks[VisionLayerTrack].self_attn.q_proj
                hook_handles.append(qryParam.register_forward_pre_hook(qry0_pre_hook))
                hook_handles.append(qryParam.register_forward_hook(qry0_forward_hook))
                #------------------------------------------------------------------------------------------------------------
                #------------------------------------------------------------------------------------------------------------
                keyParam = vision_blocks[VisionLayerTrack].self_attn.k_proj
                hook_handles.append(keyParam.register_forward_pre_hook(key0_pre_hook))
                hook_handles.append(keyParam.register_forward_hook(key0_forward_hook))
                #------------------------------------------------------------------------------------------------------------
                #------------------------------------------------------------------------------------------------------------
                valParam = vision_blocks[VisionLayerTrack].self_attn.v_proj
                hook_handles.append(valParam.register_forward_pre_hook(val0_pre_hook))
                hook_handles.append(valParam.register_forward_hook(val0_forward_hook))
                #------------------------------------------------------------------------------------------------------------
                #------------------------------------------------------------------------------------------------------------

                visOutProjParam = vision_blocks[VisionLayerTrack].self_attn.out_proj
                hook_handles.append(visOutProjParam.register_forward_pre_hook(visOutProj_pre_hook))
                hook_handles.append(visOutProjParam.register_forward_hook(visOutProj_forward_hook))

                #------------------------------------------------------------------------------------------------------------
                #------------------------------------------------------------------------------------------------------------

                # MODIFIED vs Qwen: visGate/visUp/visDown -> visFC1/visFC2 (CLIP 2-layer MLP)
                visFC1Param = vision_blocks[VisionLayerTrack].mlp.fc1
                hook_handles.append(visFC1Param.register_forward_pre_hook(visFC1_pre_hook))
                hook_handles.append(visFC1Param.register_forward_hook(visFC1_forward_hook))
                #------------------------------------------------------------------------------------------------------------
                #------------------------------------------------------------------------------------------------------------

                visFC2Param = vision_blocks[VisionLayerTrack].mlp.fc2
                hook_handles.append(visFC2Param.register_forward_pre_hook(visFC2_pre_hook))
                hook_handles.append(visFC2Param.register_forward_hook(visFC2_forward_hook))

                #------------------------------------------------------------------------------------------------------------
                #------------------------------------------------------------------------------------------------------------

                #------------ vision-to-language projector hook (final Linear of
                # multi_modal_projector, analogue of Qwen's merger.mlp[2]) -----
                MulModProjParam = mm_projector.linear_2
                hook_handles.append(MulModProjParam.register_forward_pre_hook(MulModProj_pre_hook))
                hook_handles.append(MulModProjParam.register_forward_hook(MulModProj_forward_hook))
                # ------- language hooks begin ----------------

                qryLanParam = language_layers[LanLayerTrack].self_attn.q_proj
                hook_handles.append(qryLanParam.register_forward_pre_hook(qryLan_pre_hook))
                hook_handles.append(qryLanParam.register_forward_hook(qryLan_forward_hook))


                keyLanParam = language_layers[LanLayerTrack].self_attn.k_proj
                hook_handles.append(keyLanParam.register_forward_pre_hook(keyLan_pre_hook))
                hook_handles.append(keyLanParam.register_forward_hook(keyLan_forward_hook))

                valLanParam = language_layers[LanLayerTrack].self_attn.v_proj
                hook_handles.append(valLanParam.register_forward_pre_hook(valLan_pre_hook))
                hook_handles.append(valLanParam.register_forward_hook(valLan_forward_hook))


                gate_proj = language_layers[LanLayerTrack].mlp.gate_proj # layer 0 doing great
                hook_handles.append(gate_proj.register_forward_pre_hook(gate_proj_pre_hook))
                hook_handles.append(gate_proj.register_forward_hook(gate_proj_forward_hook))

                up_proj = language_layers[LanLayerTrack].mlp.up_proj # layer 0 doing great
                hook_handles.append(up_proj.register_forward_pre_hook(up_proj_pre_hook))
                hook_handles.append(up_proj.register_forward_hook(up_proj_forward_hook))


                down_proj = language_layers[LanLayerTrack].mlp.down_proj # layer 0 doing great
                hook_handles.append(down_proj.register_forward_pre_hook(down_proj_pre_hook))
                hook_handles.append(down_proj.register_forward_hook(down_proj_forward_hook))

                allTopRightSingularVectors = [getTopRightSingularVectorForAttentioHeads(qryParam),
                                            getTopLeftSingularVectorForAttentioHeads(qryParam),

                                            getTopRightSingularVectorForAttentioHeads(keyParam),
                                            getTopLeftSingularVectorForAttentioHeads(keyParam),

                                            getTopRightSingularVectorForAttentioHeads(valParam),
                                            getTopLeftSingularVectorForAttentioHeads(valParam),

                                            getTopRightSingularVector(visOutProjParam),
                                            getTopLeftSingularVector(visOutProjParam),

                                            getTopRightSingularVector(visFC1Param),
                                            getTopLeftSingularVector(visFC1Param),

                                            getTopRightSingularVector(visFC2Param),
                                            getTopLeftSingularVector(visFC2Param),

                                            getTopRightSingularVector(MulModProjParam),
                                            getTopLeftSingularVector(MulModProjParam),

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

            # Load original image (keep original resolution)
            '''point_labels = [
            "query proj\n(vis)", "key proj\n(vis)", "value proj\n(vis)", "att output\nproj (vis)",
            "MLP gate\n(vis)", "MLP up\n(vis)", "MLP down\n(vis)", "Vis-to-lan\nproj", "query proj\n(lan)",
            "key proj\n(lan)", "value proj\n(lan)", "MLP gate\nproj(lan)", "MLP up\nproj (lan)",
            "MLP down\nproj (lan)"
            ]'''

            point_labels = [
            "query proj", "key proj", "value proj", "att output\nproj",
            "MLP fc1\n(vis)", "MLP fc2\n(vis)", "Vis-to-lan\nproj", "query proj\n",
            "key proj\n", "value proj\n", "MLP gate\nproj", "MLP up\nproj",
            "MLP down\nproj"
            ]


            if attackMode == "vis":
                # MODIFIED vs Qwen: LLaVA's vision side has 6 tracked points
                # (q, k, v, out_proj, fc1, fc2) instead of Qwen's 7.
                point_labels = point_labels[:6]
            else:
                # MODIFIED vs Qwen: language items start at index 7 (was 8).
                point_labels = point_labels[7:]

            PostAttackAlignments = []

            # MODIFIED vs Qwen: LLaVA image folder (same as llava_attack/Llava1p5FullDistancesOverlap.py)
            IMAGE_PATH = f"../interpretAttacks/gemma_attack/dataSamplesForQuant/{attackSample}.JPEG"



            pil = Image.open(IMAGE_PATH).convert("RGB")
            x_orig01 = pil_to_tensor01(pil).to(device)

            template_inputs = build_template_inputs(tokenizer, QUESTION, device)

            if device.type == "cuda":
                torch.cuda.empty_cache()
            # MODIFIED vs Qwen: LLaVA adversary paths (saved by the LLaVA attack scripts,
            # which ran from interpretAttacks/ -> "../interpretAttacks/llava_attack/..." from spectralShift/).
            # `elif` chain so that nllm keeps its own path (in the Qwen script the final
            # `if ega ... else` overwrote the nllm path with the bsa one).
            if attck_type=="bsa":
                adv_noise_path = (
                    f"../interpretAttacks/llava_attack/outputsStorage/advOutputs/{attackSample}/"
                    f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
                    f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
                    f"num_steps_{num_steps}_.pt"
                )

            elif attck_type=="nllm":
                adv_noise_path = (
                    f"../interpretAttacks/llava_attack/outputsStorage/"
                    f"advOutputs/{attackSample}/"
                    f"adv_ORIG_attackType_{attck_type}"
                    f"_lr_{lr}_eps_{epsilon}"
                    f"_num_steps_{num_steps}_.pt"
                )

            elif attck_type == "ega":
                ega_ratio = 0.2
                adv_noise_path = (
                    f"../interpretAttacks/llava_attack/outputsStorage/"
                    f"advOutputs/{attackSample}/"
                    f"adv_ORIG_attackType_{attck_type}"
                    f"_lr_{lr}_eps_{epsilon}"
                    f"_num_steps_{num_steps}"
                    f"_ratio_{ega_ratio}.pt"
                )
            else:
                # justNoise / cleanImages: a bsa delta is loaded only as a placeholder
                # (it is not used for these two attack types), same as Qwen.
                attck_typeTemp = "bsa"
                adv_noise_path = (
                    f"../interpretAttacks/llava_attack/outputsStorage/advOutputs/{attackSample}/"
                    f"adv_ORIG_attackType_{attck_typeTemp}_lr_{lr}_eps_{epsilon}_"
                    f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
                    f"num_steps_{num_steps}_.pt"
                )

            best_delta = torch.load(adv_noise_path, map_location=device).to(device=device, dtype=x_orig01.dtype) #* 0


            x_adv01, best_pert, RightSingularInputAlignmentAgainstAdversary, FlattenedAlignmentDistributionsAdversary = adam_attack_original_space(
                model=model,
                image_processor=image_processor,
                template_inputs=template_inputs,
                x_orig01=x_orig01,
                attck_type=attck_type,
                whatKindOfAdversary = "strong",
                num_steps=num_steps,
                lr=lr,
                epsilon=epsilon,
                thickEpsilon = thickEpsilon,
                device=device,
                #save_conv_path=conv_path,
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

            #print("FlattenedAlignmentDistributionsAdversary", FlattenedAlignmentDistributionsAdversary)
            #FlattenedAlignmentDistributionsAdversary = np.array(FlattenedAlignmentDistributionsAdversary.item())


            #for i in range(len(FlattenedAlignmentDistributionsAdversary)):
                #print("FlattenedAlignmentDistributionsAdversary[i].shape", FlattenedAlignmentDistributionsAdversary[i].shape)

            final = RightSingularInputAlignmentAgainstAdversary
            PostAttackAlignments.append(RightSingularInputAlignmentAgainstAdversary)
            #print("final", final)

            #------------------------------------------------------------------------------------------------------------------------------------------------------------------
            x_adv01, best_pert, RightSingularInputAlignmentAgainstOriginal, FlattenedAlignmentDistributionsOriginal = adam_attack_original_space(
                model=model,
                image_processor=image_processor,
                template_inputs=template_inputs,
                x_orig01=x_orig01,
                attck_type=attck_type,
                whatKindOfAdversary = "weakened",
                num_steps=num_steps,
                lr=lr,
                epsilon=epsilon,
                thickEpsilon = thickEpsilon,
                device=device,
                #save_conv_path=conv_path,
                AttackStartLayer = AttackStartLayer,
                numLayerstAtAtime = numLayerstAtAtime,
                allTopRightSingularVectors = allTopRightSingularVectors,
                best_delta = best_delta #best_delta + weak_delta #*0
            )

            if attackMode == "vis":
                RightSingularInputAlignmentAgainstOriginal = RightSingularInputAlignmentAgainstOriginal[:6]
                FlattenedAlignmentDistributionsOriginal = FlattenedAlignmentDistributionsOriginal[:6]

            else:
                RightSingularInputAlignmentAgainstOriginal = RightSingularInputAlignmentAgainstOriginal[7:]
                FlattenedAlignmentDistributionsOriginal = FlattenedAlignmentDistributionsOriginal[7:]


                #print("len(FlattenedAlignmentDistributionsOriginal)", len(FlattenedAlignmentDistributionsOriginal))

            # MODIFIED: third pass - weak (gaussian, same L_inf bound) adversary.
            x_adv01, best_pert, RightSingularInputAlignmentAgainstWeak, FlattenedAlignmentDistributionsWeak = adam_attack_original_space(
                model=model,
                image_processor=image_processor,
                template_inputs=template_inputs,
                x_orig01=x_orig01,
                attck_type=attck_type,
                whatKindOfAdversary = "weakened",
                num_steps=num_steps,
                lr=lr,
                epsilon=epsilon,
                thickEpsilon = thickEpsilon,
                device=device,
                AttackStartLayer = AttackStartLayer,
                numLayerstAtAtime = numLayerstAtAtime,
                allTopRightSingularVectors = allTopRightSingularVectors,
                best_delta = best_delta
            )

            if attackMode == "vis":
                RightSingularInputAlignmentAgainstWeak = RightSingularInputAlignmentAgainstWeak[:6]
                FlattenedAlignmentDistributionsWeak = FlattenedAlignmentDistributionsWeak[:6]
            else:
                RightSingularInputAlignmentAgainstWeak = RightSingularInputAlignmentAgainstWeak[7:]
                FlattenedAlignmentDistributionsWeak = FlattenedAlignmentDistributionsWeak[7:]


            DiffStrongThisSample = []
            DiffWeakThisSample = []
            for i in range(len(FlattenedAlignmentDistributionsOriginal)):
                orig_c = FlattenedAlignmentDistributionsOriginal[i]#.float()
                adv_c = FlattenedAlignmentDistributionsAdversary[i]#.float()
                weak_c = FlattenedAlignmentDistributionsWeak[i]#.float()

                token_dim = 0 if orig_c.dim() == 2 else 1

                diffStrong = (orig_c - adv_c).pow(2).mean(dim=token_dim).sqrt().flatten()
                diffWeak = (orig_c - weak_c).pow(2).mean(dim=token_dim).sqrt().flatten()


                DiffStrongThisSample.append(diffStrong)
                DiffWeakThisSample.append(diffWeak)


            for i in range(len(DiffWeakThisSample)):
                #print("FlattenedAlignmentDistributionsAdversary[i].shape", DiffStrongThisSample[i].shape)
                #print("FlattenedAlignmentDistributionsOriginal[i].shape", DiffWeakThisSample[i].shape)

                #print("FlattenedAlignmentDistributionsAdversary[i].shape", DiffStrongThisSample[i].shape)
                #print("FlattenedAlignmentDistributionsOriginal[i].shape", DiffWeakThisSample[i].shape)

                #weak = averagedAggregationOverFlattenedAlignmentDistributionsOriginal[i].detach().to(torch.float32).cpu().numpy()
                #strong = averagedAggregationFlattenedAlignmentDistributionsAdversary[i].detach().to(torch.float32).cpu().numpy()

                weak = DiffWeakThisSample[i]#.detach().to(torch.float32).cpu().numpy()
                strong = DiffStrongThisSample[i]#.detach().to(torch.float32).cpu().numpy()

                weak = (weak)
                strong = (strong)

                if torch.sum(strong==weak) / len(strong) == 1.0:
                    print("strong and weak are exactly the same")

                #print("weak", weak)
                #print("strong", strong)

                #print("strong.shape", strong.shape)
                probs = torch.sum(strong>weak) / len(strong)
                probsAlllayers.append(probs.item())

                #print("probs", probs)
                if probs >= probMax:
                    probMax = probs#.copy_()

                if probs < probMin:
                    probMin = probs#.copy_()
                    
                #curDetScore = probs * 100
                print(f"attackSample: {attackSample}, LanLayerTrack: {LanLayerTrack}, probs: {probs}, probMax: {probMax}")
                NumTimesYouHit+=1
                if probs > detectionThreshold:
                    #countAdvChance +=1
                    NumTimesYouHitTheMark+=1

                '''if probs < ignoreThreshold:
                    countIgnore +=1'''

                '''if countAdvChance > NumCountsSet:
                    break'''
                '''if countIgnore > NumCountsSet:
                    break'''

            # ADDED vs Qwen: remove this layer's hooks before moving to the next one
            for h in hook_handles:
                h.remove()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            '''if countAdvChance > NumCountsSet:
                print(f"attackSample {attackSample} is an adversary")
                print(f"Found while tracking {LanLayerTrack}")
                print()
                break
            if countIgnore > NumCountsSet:
                print(f"attackSample {attackSample} is not an adversary")
                print(f"Found while tracking {LanLayerTrack}")
                print()
                break'''

        probsAlllayersArray = np.array(probsAlllayers)

        perAttackSampleProbMaxes.append(probMax.item())
        perAttackSampleProbMins.append(probMin.item())

        print("NumTimesYouHitTheMark", NumTimesYouHitTheMark)

        print("NumTimesYouHit", NumTimesYouHit)

        hittingPercentage = NumTimesYouHitTheMark/NumTimesYouHit
        print("hittingPercentage", hittingPercentage)
        NumTimesYouHitTheMarkPerSample.append(NumTimesYouHitTheMark)

        print("perAttackSampleProbMaxes", perAttackSampleProbMaxes)
        print("perAttackSampleProbMins", perAttackSampleProbMins)
        print("NumTimesYouHitTheMarkPerSample", NumTimesYouHitTheMarkPerSample)

        probsAllSamplesAlllayer.append(probsAlllayersArray)
    probsAllSamplesAlllayerArray = np.array(probsAllSamplesAlllayer)

    print("probsAllSamplesAlllayerArray.shape", probsAllSamplesAlllayerArray.shape)
    os.makedirs("llava_attack/allProbMaxes", exist_ok=True)
    np.save(f"llava_attack/allProbMaxes/ProbMaxes_{attackMode}_attck_type_{attck_type}_epsilon_{epsilon}_thickEpsilon_{thickEpsilon}_NumattackSamples_{attackSample}_detectionThreshold_{detectionThreshold}.npy", np.array(perAttackSampleProbMaxes))
    np.save(f"llava_attack/allProbMaxes/ProbMins_{attackMode}_attck_type_{attck_type}_epsilon_{epsilon}_thickEpsilon_{thickEpsilon}_NumattackSamples_{attackSample}_detectionThreshold_{detectionThreshold}.npy", np.array(perAttackSampleProbMins))
    np.save(f"llava_attack/allProbMaxes/ChancesYouHit_{attackMode}_attck_type_{attck_type}_epsilon_{epsilon}_thickEpsilon_{thickEpsilon}_NumattackSamples_{attackSample}_detectionThreshold_{detectionThreshold}.npy", np.array(NumTimesYouHitTheMarkPerSample))
    np.save(f"llava_attack/allProbMaxes/PerSampleLayerStakedHits_{attackMode}_attck_type_{attck_type}_epsilon_{epsilon}_thickEpsilon_{thickEpsilon}_NumattackSamples_{attackSample}_detectionThreshold_{detectionThreshold}.npy", probsAllSamplesAlllayerArray)

if __name__ == "__main__":
    main()
