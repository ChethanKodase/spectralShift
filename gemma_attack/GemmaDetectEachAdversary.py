

'''


export CUDA_VISIBLE_DEVICES=0
conda activate gemma3
cd spectralShift
python gemma_attack/GemmaDetectEachAdversary.py --attck_type bsa --desired_norm_l_inf 0.005 --thickEpsilon 0.05 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan


export CUDA_VISIBLE_DEVICES=1
conda activate gemma3
cd spectralShift
python gemma_attack/GemmaDetectEachAdversary.py --attck_type nllm --desired_norm_l_inf 0.005 --thickEpsilon 0.05 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan


export CUDA_VISIBLE_DEVICES=2
conda activate gemma3
cd spectralShift
python gemma_attack/GemmaDetectEachAdversary.py --attck_type ega --desired_norm_l_inf 0.005 --thickEpsilon 0.05 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --ega_ratio 0.2


'''




import os
# MODIFIED vs Qwen/older-Gemma scripts (ported from
# qwen/Qwen2p5DetectEachAdversary.py, itself adopted from
# qwen/QwenUntargeted_BSA.py and qwen/QwenUntargeted_BSA_inference.py): this
# must be set before any CUDA/cuBLAS context is created for deterministic
# matmuls to actually take effect, so it has to come immediately after
# `import os`, before torch is imported.
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
# MODIFIED vs older Gemma scripts: ported verbatim from
# qwen/Qwen2p5DetectEachAdversary.py so this file behaves identically (same
# deterministic-algorithm / eager-attention posture) to the Qwen detector
# this script mirrors, rather than reusing the simpler set_seed/backend
# settings of the older gemma_attack/RealSpectralSubSpaceAlignment* scripts.
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
# Differentiable preprocessing (approx Gemma3)
# MODIFIED vs Qwen: Qwen2.5-VL does a dynamic-resolution resize (rounded to
# patch*merge multiples) and repacks pixels into flattened patch tokens.
# Gemma3's SigLIP image processor instead does a fixed-size resize + center
# crop, with no image_grid_thw tensor at all (necessary structural change,
# taken directly from
# gemma_attack/RealSpectralSubSpaceAlignmentPostAttackExaminerFullHistogramOverlap.py
# so the same perturbations line up).
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
    Differentiable approximation of Gemma3's image processor pipeline.
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
    MODIFIED vs Qwen (ported unchanged): Gemma3's SigLIP vision tower and its
    language model both keep a leading batch dimension of size 1, so
    `InputToLayer[0]` reliably drops that batch dim and leaves an (N, d_model)
    tensor -- unlike Qwen2.5-VL's vision transformer, which runs on an
    already-unbatched (total_tokens, hidden_dim) tensor. This helper is kept
    (rather than hard-coding `InputToLayer[0]`) so the same alignment
    functions work unchanged whichever convention a given hook point turns
    out to follow.
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

    # MODIFIED vs older Gemma scripts (ported from Qwen2p5DetectEachAdversary.py):
    # return the raw (un-reduced) coeffs instead of coeffs.mean(dim=1). We
    # need the pre-reduction coeffs so that, at the call site, we can compute
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

    # MODIFIED vs older Gemma scripts (ported from Qwen2p5DetectEachAdversary.py):
    # return the raw (un-reduced) coeffs (shape h,n,k) instead of
    # coeffs.mean(dim=1). Same reasoning as above: we need the pre-reduction
    # tensor to compute an L2 difference against another coeffs tensor first.
    # NOTE: this helper is fully generic in num_heads (it just reads V's own
    # shape), so it works unchanged for Gemma3's vision heads (SigLIP MHA, 16
    # heads) AND for the language-model query heads (8) as well as the GQA
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
# MODIFIED vs Qwen: robust getters for the vision encoder layers / language
# layers / vision-to-language projector. Gemma3ForConditionalGeneration
# exposes these directly as model.vision_tower.vision_model.encoder.layers
# and model.language_model.model.layers (no extra ".model." wrapper the way
# some transformers versions add for Qwen2.5-VL), but we still defensively
# probe with hasattr() so this keeps working across transformers versions,
# mirroring the same pattern used in
# qwen/Qwen2p5DetectEachAdversary.py's get_vision_module_and_blocks() /
# get_language_module_and_layers().
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
def get_vision_module_and_blocks(model):
    if hasattr(model, "vision_tower") and hasattr(model.vision_tower, "vision_model") and hasattr(model.vision_tower.vision_model.encoder, "layers"):
        return model.vision_tower.vision_model, model.vision_tower.vision_model.encoder.layers
    elif hasattr(model, "model") and hasattr(model.model, "vision_tower") and hasattr(model.model.vision_tower.vision_model.encoder, "layers"):
        return model.model.vision_tower.vision_model, model.model.vision_tower.vision_model.encoder.layers
    else:
        raise RuntimeError("Could not find Gemma3 vision encoder layers.")


def get_language_module_and_layers(model):
    if hasattr(model, "language_model") and hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
        return model.language_model.model, model.language_model.model.layers
    elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model, model.language_model.layers
    elif hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model, model.model.language_model.layers
    else:
        raise RuntimeError("Could not find Gemma3 language-model layers.")


def get_multi_modal_projector(model):
    if hasattr(model, "multi_modal_projector"):
        return model.multi_modal_projector
    elif hasattr(model, "model") and hasattr(model.model, "multi_modal_projector"):
        return model.model.multi_modal_projector
    else:
        raise RuntimeError("Could not find Gemma3 multi_modal_projector module.")


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

# MODIFIED vs Qwen: Qwen2.5-VL's vision MLP is a 3-projection SwiGLU block
# (gate_proj/up_proj/down_proj). Gemma3's SigLIP vision MLP is a plain
# 2-layer FC (fc1/fc2), exactly like the older gemma_attack reference
# scripts, so we keep the FC1/FC2 hooks instead of introducing a
# visGate/visUp/visDown split that has no counterpart in this architecture.
FC1_inputs = {}
def FC1_pre_hook(module, inputs):
    FC1_inputs["FC1_in"] = inputs[0]
FC1_outputs = {}
def FC1_forward_hook(module, inputs, output):
    FC1_outputs["FC1_out"] = output
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
        pv_clean_fixed = gemma_preprocess_differentiable(x_orig01, processor)

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
        # MODIFIED vs Qwen: use the robust getters + config-based head counts
        # (Gemma3's vision attention is plain MHA; the language model uses
        # GQA, so num_key_value_heads is tracked separately).
        vision_module, vision_layers = get_vision_module_and_blocks(model)
        vision_cfg = _resolve_vision_config(model)
        text_cfg = _resolve_text_config(model)

        num_heads = getattr(vision_cfg, "num_attention_heads", getattr(vision_cfg, "num_heads", 16))
        d_model = getattr(vision_cfg, "hidden_size", vision_layers[0].self_attn.q_proj.weight.shape[1])
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



    x_adv01 = (x_orig01 + delta).clamp(0.0, 1.0)
    x_adv01 = torch.max(torch.min(x_adv01, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

    if whatKindOfAdversary == "weakened":
        #thickEpsilon = 0.05
        weak_delta = torch.randn_like(best_delta) * thickEpsilon
        x_adv01 = (x_adv01 + weak_delta).clamp(0.0, 1.0)
        x_adv01 = torch.max(torch.min(x_adv01, x_orig01 + thickEpsilon), x_orig01 - thickEpsilon).clamp(0.0, 1.0)


    # preprocess adv (must be differentiable)
    pv_adv = gemma_preprocess_differentiable(x_adv01, processor)

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

        # MODIFIED vs Qwen: FC1 (Gemma3 vision MLP, plain 2-layer FC) instead
        # of visGate (Qwen vision SwiGLU MLP gate_proj).
        InputToLayer = FC1_inputs.get("FC1_in")
        FC1_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[8])
        RightSingularInputAlignment.append(FC1_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # MODIFIED vs Qwen: FC2 (Gemma3 vision MLP) instead of visDown (Qwen
        # vision SwiGLU MLP down_proj) -- Gemma3's vision MLP has only these
        # two projections, unlike Qwen's 3-projection SwiGLU block, so there
        # is no visUp counterpart here.
        InputToLayer = FC2_inputs.get("FC2_in")
        FC2_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[10])
        RightSingularInputAlignment.append(FC2_in_MAE)
        AlignmentDistributions.append(ProjDistrib)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        InputToLayer = MulModProj_inputs.get("MulModProj_in")
        MulModProj_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[12])
        #OutputOfLayer = MulModProj_outputs.get("MulModProj_out")
        #MulModProj_out_MAE = getMeanAlignmentWithTopLeftSingularVector(OutputOfLayer, allTopRightSingularVectors[13])
        #print("MulModProj_in_MAE MulModProj_out_MAE", MulModProj_in_MAE, MulModProj_out_MAE)
        RightSingularInputAlignment.append(MulModProj_in_MAE)
        AlignmentDistributions.append(ProjDistrib)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        InputToLayer = qryLan_inputs.get("qryLan_in")
        qryLan_in_MAE, ProjDistrib = getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, allTopRightSingularVectors[14])
        #OutputOfLayer = qryLan_outputs.get("qryLan_out")
        #qryLan_out_MAE = getMeanAlignmentWithAttentionHeadTopLeftSingularVector(OutputOfLayer, num_headsT, allTopRightSingularVectors[15])
        #print("qryLan_in_MAE qryLan_out_MAE", qryLan_in_MAE, qryLan_out_MAE)
        RightSingularInputAlignment.append(qryLan_in_MAE)
        AlignmentDistributions.append(ProjDistrib)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        InputToLayer = keyLan_inputs.get("keyLan_in")
        keyLan_in_MAE, ProjDistrib = getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, allTopRightSingularVectors[16])
        #OutputOfLayer = keyLan_outputs.get("keyLan_out")
        #keyLan_out_MAE = getMeanAlignmentWithAttentionHeadTopLeftSingularVector(OutputOfLayer, num_kv_headsT, allTopRightSingularVectors[17])
        #print("keyLan_in_MAE keyLan_out_MAE", keyLan_in_MAE, keyLan_out_MAE)
        RightSingularInputAlignment.append(keyLan_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        InputToLayer = valLan_inputs.get("valLan_in")
        valLan_in_MAE, ProjDistrib = getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, allTopRightSingularVectors[18])
        #OutputOfLayer = valLan_outputs.get("valLan_out")
        #valLan_out_MAE = getMeanAlignmentWithAttentionHeadTopLeftSingularVector(OutputOfLayer, num_kv_headsT, allTopRightSingularVectors[19])
        #print(" valLan_in_MAE,  valLan_out_MAE ", valLan_in_MAE,  valLan_out_MAE)
        RightSingularInputAlignment.append(valLan_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        InputToLayer = GateLayerInputs.get("gate_in")
        gate_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[20])
        #OutputOfLayer = GateLayerOutputs.get("gate_out")
        #gate_out_MAE = getMeanAlignmentWithTopLeftSingularVector(OutputOfLayer, allTopRightSingularVectors[21])
        #print("gate_in_MAE, gate_out_MAE", gate_in_MAE, gate_out_MAE)
        RightSingularInputAlignment.append(gate_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        InputToLayer = UpLayerInputs.get("up_in")
        up_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[22])
        #OutputOfLayer = UpLayerOutputs.get("up_out")
        #up_out_MAE = getMeanAlignmentWithTopLeftSingularVector(OutputOfLayer, allTopRightSingularVectors[23])
        #print("up_out, up_out_MAE", up_in_MAE, up_out_MAE)
        RightSingularInputAlignment.append(up_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        InputToLayer = DownLayer_inputs.get("down_in")
        down_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[24])
        #OutputOfLayer = DownLayer_outputs.get("down_out")
        #down_out_MAE = getMeanAlignmentWithTopLeftSingularVector(OutputOfLayer, allTopRightSingularVectors[25])
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
    parser = argparse.ArgumentParser(description="Gemma3 ORIGINAL-image-space adversarial attack (no squeeze)")
    parser.add_argument("--attck_type", type=str, default="bsa",
                        help="bsa | nllm | ega")
    parser.add_argument("--desired_norm_l_inf", type=float, default=0.03,
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
    # MODIFIED vs Qwen: ega_ratio is a Gemma-only knob (mirrors the EGA
    # perturbation-file naming used by the Gemma attack generators), so it is
    # exposed as a CLI arg here even though Qwen2p5DetectEachAdversary.py
    # hard-codes it locally.
    parser.add_argument("--ega_ratio", type=float, default=0.2,
                    help="ega_ratio used when locating an EGA adv_noise_path")



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

    ega_ratio = float(args.ega_ratio)


    MODEL_PATH = "../illcond/gemma_attack/Gemma3-4b"
    QUESTION = "What is shown in this image?"
    MAX_NEW_TOKENS = 128


    #attackMode = "lan"
    #attackMode = "vis"

    #IMAGE_PATH = f"gemma_attack/dataSamples/interference68.jpeg"


    os.makedirs("gemma_attack/outputsStorageImagenet", exist_ok=True)
    os.makedirs("gemma_attack/outputsStorageImagenet/advOutputs", exist_ok=True)
    os.makedirs("gemma_attack/allProbMaxes", exist_ok=True)

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
    vision_module, vision_layers = get_vision_module_and_blocks(model)
    language_module, language_layers = get_language_module_and_layers(model)

    vision_cfg = _resolve_vision_config(model)
    text_cfg = _resolve_text_config(model)

    num_heads = getattr(vision_cfg, "num_attention_heads", getattr(vision_cfg, "num_heads", 16))
    d_model = getattr(vision_cfg, "hidden_size", vision_layers[0].self_attn.q_proj.weight.shape[1])
    d_head = d_model // num_heads
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

    # MODIFIED vs Qwen: Gemma3's vision-to-language multi_modal_projector
    # stores a raw (transposed) mm_input_projection_weight nn.Parameter --
    # unlike Qwen2.5-VL's "merger", which is a plain
    # nn.Sequential(Linear, GELU, Linear) whose last Linear can be SVD'd
    # directly. So we keep the Gemma-specific
    # getTopRightSingularVectorForMMproj()/getTopLeftSingularVectorForMMproj()
    # transpose-special-case helpers (as in the older gemma_attack reference
    # scripts) instead of Qwen's plain getTopRightSingularVector() on
    # merger.mlp[2].
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

    d_modelT = getattr(text_cfg, "hidden_size")
    num_headsT = getattr(text_cfg, "num_attention_heads")
    # MODIFIED vs Qwen (ported unchanged): Gemma3-4B's language model uses
    # Grouped-Query Attention (num_key_value_heads < num_attention_heads:
    # here 4 kv heads vs 8 query heads) AND decouples head_dim from
    # hidden_size/num_heads (head_dim=256, hidden_size=2560, so
    # 2560/8=320 != 256). k_proj/v_proj must therefore be split using
    # num_kv_headsT, not num_headsT, and d_head must be derived from each
    # projection's own output shape (param.shape[0] // num_heads_local)
    # rather than d_modelT // num_heads_local -- this is a necessary
    # correctness fix vs. the older gemma_attack reference scripts, which
    # reused a single num_headsT constant for q/k/v alike (see
    # gemma_attack/GemmaUntargted_ChosenSingulaeVectors.py for the same fix
    # applied to the SVD-basis-only variant of this script).
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


    perAttackSampleProbMaxes = []
    for attackSample in range(1,51):
    #for attackSample in range(1,51):
        countAdvChance = 0
        probMax = 0
        #attackSample = 39 # check
        # MODIFIED vs Qwen: Gemma3-4B has len(language_layers) == 34 decoder
        # layers (vs Qwen2.5-7B's fixed 28), so this is driven off the actual
        # number of layers found on the loaded model instead of a hard-coded
        # range(28).
        for LanLayerTrack in range(len(language_layers)):
        #for LanLayerTrack in range(2):

            with torch.no_grad():

                # ------- vision hooks begin ----------------
                # MODIFIED vs Qwen: Gemma3's SigLIP vision attention stores
                # separate q_proj/k_proj/v_proj/out_proj Linears (unlike
                # Qwen2.5-VL's single fused qkv Linear), so we hook each
                # projection directly instead of slicing a fused qkv weight.
                qryParam = vision_layers[VisionLayerTrack].self_attn.q_proj
                hook_handle = qryParam.register_forward_pre_hook(qry0_pre_hook)
                hook_handle = qryParam.register_forward_hook(qry0_forward_hook)
                #------------------------------------------------------------------------------------------------------------
                #------------------------------------------------------------------------------------------------------------
                keyParam = vision_layers[VisionLayerTrack].self_attn.k_proj
                hook_handle = keyParam.register_forward_pre_hook(key0_pre_hook)
                hook_handle = keyParam.register_forward_hook(key0_forward_hook)
                #------------------------------------------------------------------------------------------------------------
                #------------------------------------------------------------------------------------------------------------
                valParam = vision_layers[VisionLayerTrack].self_attn.v_proj
                hook_handle = valParam.register_forward_pre_hook(val0_pre_hook)
                hook_handle = valParam.register_forward_hook(val0_forward_hook)
                #------------------------------------------------------------------------------------------------------------
                #------------------------------------------------------------------------------------------------------------

                visOutProjParam = vision_layers[VisionLayerTrack].self_attn.out_proj
                hook_handle = visOutProjParam.register_forward_pre_hook(visOutProj_pre_hook)
                hook_handle = visOutProjParam.register_forward_hook(visOutProj_forward_hook)

                #------------------------------------------------------------------------------------------------------------
                #------------------------------------------------------------------------------------------------------------

                # MODIFIED vs Qwen: FC1/FC2 (Gemma3 vision MLP, plain 2-layer FC)
                FC1Param = vision_layers[VisionLayerTrack].mlp.fc1
                hook_handle = FC1Param.register_forward_pre_hook(FC1_pre_hook)
                hook_handle = FC1Param.register_forward_hook(FC1_forward_hook)
                #------------------------------------------------------------------------------------------------------------
                #------------------------------------------------------------------------------------------------------------

                FC2Param = vision_layers[VisionLayerTrack].mlp.fc2
                hook_handle = FC2Param.register_forward_pre_hook(FC2_pre_hook)
                hook_handle = FC2Param.register_forward_hook(FC2_forward_hook)

                #------------------------------------------------------------------------------------------------------------
                #------------------------------------------------------------------------------------------------------------

                #------------ vision-to-language multi_modal_projector hook -----
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


                gate_proj = language_layers[LanLayerTrack].mlp.gate_proj # layer 0 doing great
                hook_handle = gate_proj.register_forward_pre_hook(gate_proj_pre_hook)
                hook_handle = gate_proj.register_forward_hook(gate_proj_forward_hook)

                up_proj = language_layers[LanLayerTrack].mlp.up_proj # layer 0 doing great
                hook_handle = up_proj.register_forward_pre_hook(up_proj_pre_hook)
                hook_handle = up_proj.register_forward_hook(up_proj_forward_hook)


                down_proj = language_layers[LanLayerTrack].mlp.down_proj # layer 0 doing great
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

            # Load original image (keep original resolution)
            '''point_labels = [
            "query proj\n(vis)", "key proj\n(vis)", "value proj\n(vis)", "att output\nproj (vis)",
            "MLP fc1\n(vis)", "MLP fc2\n(vis)", "Vis-to-lan\nproj", "query proj\n(lan)",
            "key proj\n(lan)", "value proj\n(lan)", "MLP gate\nproj(lan)", "MLP up\nproj (lan)",
            "MLP down\nproj (lan)"
            ]'''

            # MODIFIED vs Qwen: 13 tracked points instead of 14 -- Gemma3's
            # vision MLP contributes only 2 hook points (FC1/FC2) instead of
            # Qwen's 3 (visGate/visUp/visDown).
            point_labels = [
            "query proj", "key proj", "value proj", "att output\nproj",
            "MLP fc1\n(vis)", "MLP fc2\n(vis)", "Vis-to-lan\nproj", "query proj\n",
            "key proj\n", "value proj\n", "MLP gate\nproj", "MLP up\nproj",
            "MLP down\nproj"
            ]


            if attackMode == "vis":
                # MODIFIED vs Qwen: Gemma's vision side has 6 tracked points
                # (query/key/value/att-out-proj/FC1/FC2), so we drop the last
                # one (the vis-to-lan connector, which is layer-independent)
                # leaving the first 6.
                point_labels = point_labels[:6]
            else:
                # MODIFIED vs Qwen: language items now start at index 7 (was 8).
                point_labels = point_labels[7:]

            PostAttackAlignments = []

            # Per user instructions: sample images live under
            # ../interpretAttacks/gemma_attack/dataSamplesForQuant/.
            IMAGE_PATH = f"../interpretAttacks/gemma_attack/dataSamplesForQuant/{attackSample}.JPEG"

            pil = Image.open(IMAGE_PATH).convert("RGB")
            x_orig01 = pil_to_tensor01(pil).to(device)

            template_inputs = build_template_inputs(processor, QUESTION, pil, device)

            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Per user instructions: BSA/NLLM perturbations live under
            # gemma_attack/outputsStorageImagenet/advOutputs/<sample>/, EGA
            # perturbations carry an extra ratio_{ega_ratio} suffix. interpretAttacks/gemma_attack
            if attck_type=="bsa":
                adv_noise_path = (
                    f"../interpretAttacks/gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
                    f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.pt"
                )

            if attck_type=="nllm":
                adv_noise_path = (
                    f"../interpretAttacks/gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
                    f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.pt"
                )

            if attck_type == "ega":
                adv_noise_path = (
                    f"../interpretAttacks/gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
                    f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_ratio_{ega_ratio}.pt"
                )


            best_delta = torch.load(adv_noise_path, map_location=device).to(device=device, dtype=x_orig01.dtype) * 0


            x_adv01, best_pert, RightSingularInputAlignmentAgainstAdversary, FlattenedAlignmentDistributionsAdversary = adam_attack_original_space(
                model=model,
                processor=processor,
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
                processor=processor,
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
                processor=processor,
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
                orig_c = FlattenedAlignmentDistributionsOriginal[i]
                adv_c = FlattenedAlignmentDistributionsAdversary[i]
                weak_c = FlattenedAlignmentDistributionsWeak[i]

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


                #print("weak.shape", weak.shape)

                #print("strong.shape", strong.shape)
                probs = torch.sum(strong>weak) / len(strong)
                if probs > probMax:
                    probMax = probs#.copy_()
                #curDetScore = probs * 100
                print(f"attackSample: {attackSample}, LanLayerTrack: {LanLayerTrack}, probs: {probs}, probMax: {probMax}")

                if probs > 0.9:
                    countAdvChance +=1

                if countAdvChance > 4:
                    break


            if countAdvChance > 4:
                print(f"attackSample {attackSample} is an adversary")
                print(f"Found while tracking {LanLayerTrack}")
                print()
                break
        perAttackSampleProbMaxes.append(probMax.item())

        print("perAttackSampleProbMaxes", perAttackSampleProbMaxes)

        os.makedirs("gemma_attack/allProbMaxes", exist_ok=True)
        np.save(f"gemma_attack/allProbMaxes/perAttackSampleProbMaxes_{attackMode}_attck_type_{attck_type}_epsilon_{epsilon}_thickEpsilon_{thickEpsilon}_NumattackSamples_{attackSample}_.npy", np.array(perAttackSampleProbMaxes))


if __name__ == "__main__":
    main()
