

'''

export CUDA_VISIBLE_DEVICES=0
conda deactivate
cd spectralShift/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
python qwen/Qwen2p5FullHistogramQuantitativeLanSSGRA.py --attck_type saa_loop --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --attackMode lan

'''



import os
# MODIFIED vs Gemma (adopted from qwen/QwenUntargeted_BSA.py and
# qwen/QwenUntargeted_BSA_inference.py): this must be set before any CUDA/cuBLAS
# context is created for deterministic matmuls to actually take effect, so it
# has to come immediately after `import os`, before torch is imported.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import sys
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import wasserstein_distance

from scipy.spatial.distance import jensenshannon
import pandas as pd
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
    Differentiable approximation of Qwen2.5-VL's image processor pipeline.
    Produces (pixel_values, image_grid_thw) exactly like the real processor.
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

    coeffs = H_hat @ V_hat.T        # (N, k)

    # Energy per token (sum of squared coefficients)
    per_token_energy = (coeffs ** 2).sum(dim=1)  # (N,)

    #print("coeffs.shape", coeffs.shape)
    #print("coeffs.mean(dim=1).shape", coeffs.mean(dim=1).shape)

    #coeffHIst = coeffs.mean(dim=1)

    # Mean across tokens
    mean_energy = per_token_energy.mean().item()

    # MODIFIED vs Gemma (necessary fix): average over the TOKEN axis, which
    # is dim=0 for this (N, k) coeffs tensor, not dim=1. Gemma's original
    # `coeffs.mean(dim=1)` happened to average over k (the singular-vector
    # index) here, which "worked" only because every one of Gemma's 50
    # samples produces the exact same fixed 896x896 canvas and therefore the
    # same N. Qwen2.5-VL uses native dynamic-resolution patching, so N
    # (vision tokens, and the total multimodal sequence length for language
    # hooks) is different for every image. Averaging over the true token
    # axis (dim=0) makes the returned distribution a fixed, architecture-
    # defined length (k = in_dim of this weight matrix) that no longer
    # depends on how many tokens a given image produced, so the cross-sample
    # torch.stack(...) later in main() is valid regardless of resolution.
    return mean_energy, coeffs.mean(dim=0)

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

    # NOTE: unlike the plain-Linear helper above, `dim=1` here already IS the
    # token axis (coeffs shape is (num_heads, N, k)), so this average over
    # tokens is correct as-is and does not depend on N -- no change needed
    # for Qwen's variable per-image token count.
    return mean_energy_all, coeffs.mean(dim=1)


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

def center_of_mass_shift(orig, adv):
    ranks = np.arange(len(orig))
    w_orig = orig**2
    w_adv = adv**2
    com_orig = np.sum(ranks * w_orig) / np.sum(w_orig)
    com_adv = np.sum(ranks * w_adv) / np.sum(w_adv)
    shift = com_adv - com_orig

    return com_orig, com_adv, shift



def mean_alignment_shift(orig, adv):
    mean_orig = np.mean(orig)
    mean_adv = np.mean(adv)
    delta_mean = mean_adv - mean_orig
    return mean_orig, mean_adv, delta_mean

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
        pv_clean_fixed, grid_clean_fixed = qwen_preprocess_differentiable(x_orig01, processor)

        clean_inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
        clean_inputs["pixel_values"] = pv_clean_fixed
        clean_inputs["image_grid_thw"] = grid_clean_fixed
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
        # MODIFIED vs Gemma: use the robust getters + config-based head counts
        # (Qwen2.5-VL vision attention is plain MHA; the language model uses
        # GQA, so num_key_value_heads is tracked separately).
        vision_module, vision_blocks = get_vision_module_and_blocks(model)
        vision_cfg = _resolve_vision_config(model)
        text_cfg = _resolve_text_config(model)

        num_heads = getattr(vision_cfg, "num_heads", 16)
        d_model = getattr(vision_cfg, "hidden_size", vision_blocks[0].attn.qkv.weight.shape[1])
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

    # preprocess adv (must be differentiable)
    pv_adv, grid_adv = qwen_preprocess_differentiable(x_adv01, processor)

    adv_inputs["pixel_values"] = pv_adv
    adv_inputs["image_grid_thw"] = grid_adv

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

        # MODIFIED vs Gemma: FC1 -> visGate (Qwen vision SwiGLU MLP gate_proj)
        InputToLayer = visGate_inputs.get("visGate_in")
        visGate_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[8])
        RightSingularInputAlignment.append(visGate_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # ADDED vs Gemma: visUp (Qwen vision SwiGLU MLP up_proj) -- Qwen's
        # vision MLP has a 3rd projection Gemma's 2-layer FC does not have.
        InputToLayer = visUp_inputs.get("visUp_in")
        visUp_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[10])
        RightSingularInputAlignment.append(visUp_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # MODIFIED vs Gemma: FC2 -> visDown (Qwen vision SwiGLU MLP down_proj)
        InputToLayer = visDown_inputs.get("visDown_in")
        visDown_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[12])
        RightSingularInputAlignment.append(visDown_in_MAE)
        AlignmentDistributions.append(ProjDistrib)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        InputToLayer = MulModProj_inputs.get("MulModProj_in")
        MulModProj_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[14])
        #OutputOfLayer = MulModProj_outputs.get("MulModProj_out")
        #MulModProj_out_MAE = getMeanAlignmentWithTopLeftSingularVector(OutputOfLayer, allTopRightSingularVectors[15])
        #print("MulModProj_in_MAE MulModProj_out_MAE", MulModProj_in_MAE, MulModProj_out_MAE)
        RightSingularInputAlignment.append(MulModProj_in_MAE)
        AlignmentDistributions.append(ProjDistrib)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        InputToLayer = qryLan_inputs.get("qryLan_in")
        qryLan_in_MAE, ProjDistrib = getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, allTopRightSingularVectors[16])
        #OutputOfLayer = qryLan_outputs.get("qryLan_out")
        #qryLan_out_MAE = getMeanAlignmentWithAttentionHeadTopLeftSingularVector(OutputOfLayer, num_headsT, allTopRightSingularVectors[17])
        #print("qryLan_in_MAE qryLan_out_MAE", qryLan_in_MAE, qryLan_out_MAE)
        RightSingularInputAlignment.append(qryLan_in_MAE)
        AlignmentDistributions.append(ProjDistrib)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        InputToLayer = keyLan_inputs.get("keyLan_in")
        keyLan_in_MAE, ProjDistrib = getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, allTopRightSingularVectors[18])
        #OutputOfLayer = keyLan_outputs.get("keyLan_out")
        #keyLan_out_MAE = getMeanAlignmentWithAttentionHeadTopLeftSingularVector(OutputOfLayer, num_kv_headsT, allTopRightSingularVectors[19])
        #print("keyLan_in_MAE keyLan_out_MAE", keyLan_in_MAE, keyLan_out_MAE)
        RightSingularInputAlignment.append(keyLan_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        InputToLayer = valLan_inputs.get("valLan_in")
        valLan_in_MAE, ProjDistrib = getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, allTopRightSingularVectors[20])
        #OutputOfLayer = valLan_outputs.get("valLan_out")
        #valLan_out_MAE = getMeanAlignmentWithAttentionHeadTopLeftSingularVector(OutputOfLayer, num_kv_headsT, allTopRightSingularVectors[21])
        #print(" valLan_in_MAE,  valLan_out_MAE ", valLan_in_MAE,  valLan_out_MAE)
        RightSingularInputAlignment.append(valLan_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        InputToLayer = GateLayerInputs.get("gate_in")
        gate_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[22])
        #OutputOfLayer = GateLayerOutputs.get("gate_out")
        #gate_out_MAE = getMeanAlignmentWithTopLeftSingularVector(OutputOfLayer, allTopRightSingularVectors[23])
        #print("gate_in_MAE, gate_out_MAE", gate_in_MAE, gate_out_MAE)
        RightSingularInputAlignment.append(gate_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        InputToLayer = UpLayerInputs.get("up_in")
        up_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[24])
        #OutputOfLayer = UpLayerOutputs.get("up_out")
        #up_out_MAE = getMeanAlignmentWithTopLeftSingularVector(OutputOfLayer, allTopRightSingularVectors[25])
        #print("up_out, up_out_MAE", up_in_MAE, up_out_MAE)
        RightSingularInputAlignment.append(up_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        InputToLayer = DownLayer_inputs.get("down_in")
        down_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[26])
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
        FlattenedAlignmentDistributions.append(AlignmentDistributions[i].flatten())
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

    #parser.add_argument("--VisionLayerTrack", type=int, default=2,
    #                    help="whcih vision layer you want to talk")
    #parser.add_argument("--LanLayerTrack", type=int, default=2,
    #                    help="whcih language layer you want to talk")
    #parser.add_argument("--kthSingVec", type=int, default=0,
    #                help="Amonhg the k singular vectors which one do you wanty")
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




    attackMode = str(args.attackMode)


    MODEL_PATH = "../illcond/QwenAttack/Qwen2.5-VL-7B-Instruct"
    QUESTION = "What is shown in this image?"
    MAX_NEW_TOKENS = 128




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



    def getTopRightSingularVector(down_proj):
        return torch.linalg.svd(down_proj.weight.to(torch.float32))[2][:]

    def getTopLeftSingularVector(down_proj):
        return torch.linalg.svd(down_proj.weight.to(torch.float32))[0][:]

    # MODIFIED vs Gemma: Qwen2.5-VL's vision-to-language "merger" is a plain
    # nn.Sequential(Linear, GELU, Linear) -- unlike Gemma's multi_modal_projector,
    # which stores a raw (transposed) mm_input_projection_weight parameter.
    # Because merger.mlp[2] is a standard nn.Linear, we hook and SVD it exactly
    # like FC1/FC2/gate/up/down below, so the Gemma-only
    # getTopRightSingularVectorForMMproj()/getTopLeftSingularVectorForMMproj()
    # transpose-special-case helpers are not needed here.

    def getTopRightSingularVectorForAttentioHeads(qryParam):
        param = qryParam.weight.to(torch.float32)
        param_heads = param.view(num_heads, d_head, d_model)
        query_vh_per_head = []
        for h in range(num_heads):
            Wh = param_heads[h]
            U, S, Vh = torch.linalg.svd(Wh.to(torch.float32) )
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
            query_vh_per_head.append(U[:])
        query_vh_per_head = torch.stack(query_vh_per_head, 0)
        return query_vh_per_head

    # ADDED vs Gemma: Qwen2.5-VL's vision attention stores a single FUSED
    # qkv Linear (weight shape (3*d_model, d_model)) instead of separate
    # q_proj/k_proj/v_proj Linears like Gemma/SigLIP. We slice out the
    # requested block first, then reuse the exact same per-head SVD logic
    # as getTopRightSingularVectorForAttentioHeads/Left above.
    def getTopRightSingularVectorForVisionQKV(qkvParam, which):
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

    def getTopLeftSingularVectorForVisionQKV(qkvParam, which):
        full_weight = qkvParam.weight.to(torch.float32)
        idx = {"q": 0, "k": 1, "v": 2}[which]
        weight_slice = full_weight[idx * d_model:(idx + 1) * d_model, :]
        weight_slice = weight_slice.view(num_heads, d_head, d_model)
        u_per_head = []
        for h in range(num_heads):
            Wh = weight_slice[h]
            U, S, Vh = torch.linalg.svd(Wh.to(torch.float32))
            u_per_head.append(U[:])
        return torch.stack(u_per_head, 0)

    #print("num_headsT", num_headsT)
    def getTopRightSingularVectorForLanAttentioHeads(qryParam, num_heads_local):
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

    def getTopLeftSingularVectorForLanAttentioHeads(qryParam, num_heads_local):
        param = qryParam.weight.to(torch.float32)
        d_headT = param.shape[0] // num_heads_local
        param_heads = param.view(num_heads_local, d_headT, d_modelT)
        query_vh_per_head = []
        for h in range(num_heads_local):
            Wh = param_heads[h]
            U, S, Vh = torch.linalg.svd(Wh.to(torch.float32) )
            query_vh_per_head.append(U[:])

        query_vh_per_head = torch.stack(query_vh_per_head, 0)
        return query_vh_per_head


    d_modelT = getattr(text_cfg, "hidden_size")
    num_headsT = getattr(text_cfg, "num_attention_heads")
    # MODIFIED vs Gemma: Qwen2.5-7B-Instruct's language model uses
    # Grouped-Query Attention (num_key_value_heads < num_attention_heads).
    # k_proj/v_proj must therefore be split using num_kv_headsT, not
    # num_headsT, otherwise the per-head reshape below is not an integer
    # split. This is a necessary correctness fix vs. the Gemma script, which
    # reused a single num_headsT constant for q/k/v alike.
    num_kv_headsT = getattr(text_cfg, "num_key_value_heads", num_headsT)


    #VisionLayerTrack = int(args.VisionLayerTrack)
    #sLanLayerTrack = int(args.LanLayerTrack)

    # MODIFIED vs Gemma: distinct output directory/filename for the Qwen
    # sweep so it never collides with the Gemma CSVs, and the directory is
    # created here (via os.makedirs) since it does not exist by default.
    csv_dir = "qwen/QuantitativeAlignmentShiftSSGRA"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "alignment_shift_sweepLan_qwen.csv")
    write_header = not os.path.exists(csv_path)

    # MODIFIED vs Gemma: Qwen2.5-7B-Instruct's language model has 28 layers
    # (0..27, see qwen/model_parameters.txt), vs Gemma's 27, so the sweep
    # range is adjusted accordingly.
    for LanLayerTrack in range(28):
        VisionLayerTrack = 0

        with torch.no_grad():

            # ------- vision hooks begin ----------------
            # MODIFIED vs Gemma: Qwen fuses q/k/v into a single Linear
            # (attn.qkv). We register the SAME pre-hook module three times
            # (once per logical q/k/v "channel") so the rest of the code
            # (qry0_inputs/key0_inputs/val0_inputs) stays structurally
            # identical to the Gemma version.
            qkvParam = vision_blocks[VisionLayerTrack].attn.qkv
            hook_handle = qkvParam.register_forward_pre_hook(qry0_pre_hook)
            hook_handle = qkvParam.register_forward_hook(qry0_forward_hook)
            #------------------------------------------------------------------------------------------------------------
            #------------------------------------------------------------------------------------------------------------
            hook_handle = qkvParam.register_forward_pre_hook(key0_pre_hook)
            hook_handle = qkvParam.register_forward_hook(key0_forward_hook)
            #------------------------------------------------------------------------------------------------------------
            #------------------------------------------------------------------------------------------------------------
            hook_handle = qkvParam.register_forward_pre_hook(val0_pre_hook)
            hook_handle = qkvParam.register_forward_hook(val0_forward_hook)
            #------------------------------------------------------------------------------------------------------------
            #------------------------------------------------------------------------------------------------------------

            visOutProjParam = vision_blocks[VisionLayerTrack].attn.proj
            hook_handle = visOutProjParam.register_forward_pre_hook(visOutProj_pre_hook)
            hook_handle = visOutProjParam.register_forward_hook(visOutProj_forward_hook)

            #------------------------------------------------------------------------------------------------------------
            #------------------------------------------------------------------------------------------------------------

            # MODIFIED vs Gemma: FC1/FC2 -> visGate/visUp/visDown (SwiGLU MLP)
            visGateParam = vision_blocks[VisionLayerTrack].mlp.gate_proj
            hook_handle = visGateParam.register_forward_pre_hook(visGate_pre_hook)
            hook_handle = visGateParam.register_forward_hook(visGate_forward_hook)
            #------------------------------------------------------------------------------------------------------------
            #------------------------------------------------------------------------------------------------------------

            visUpParam = vision_blocks[VisionLayerTrack].mlp.up_proj
            hook_handle = visUpParam.register_forward_pre_hook(visUp_pre_hook)
            hook_handle = visUpParam.register_forward_hook(visUp_forward_hook)
            #------------------------------------------------------------------------------------------------------------
            #------------------------------------------------------------------------------------------------------------

            visDownParam = vision_blocks[VisionLayerTrack].mlp.down_proj
            hook_handle = visDownParam.register_forward_pre_hook(visDown_pre_hook)
            hook_handle = visDownParam.register_forward_hook(visDown_forward_hook)

            #------------------------------------------------------------------------------------------------------------
            #------------------------------------------------------------------------------------------------------------

            #------------ vision-to-language merger hook (hook the final
            # Linear inside merger.mlp so the captured input dim matches its
            # weight) -----
            MulModProjParam = vision_module.merger.mlp[2]
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

            allTopRightSingularVectors = [getTopRightSingularVectorForVisionQKV(qkvParam, "q"),
                                        getTopLeftSingularVectorForVisionQKV(qkvParam, "q"),

                                        getTopRightSingularVectorForVisionQKV(qkvParam, "k"),
                                        getTopLeftSingularVectorForVisionQKV(qkvParam, "k"),

                                        getTopRightSingularVectorForVisionQKV(qkvParam, "v"),
                                        getTopLeftSingularVectorForVisionQKV(qkvParam, "v"),

                                        getTopRightSingularVector(visOutProjParam),
                                        getTopLeftSingularVector(visOutProjParam),

                                        getTopRightSingularVector(visGateParam),
                                        getTopLeftSingularVector(visGateParam),

                                        getTopRightSingularVector(visUpParam),
                                        getTopLeftSingularVector(visUpParam),

                                        getTopRightSingularVector(visDownParam),
                                        getTopLeftSingularVector(visDownParam),

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
        "MLP gate\n(vis)", "MLP up\n(vis)", "MLP down\n(vis)", "Vis-to-lan\nproj", "query proj\n",
        "key proj\n", "value proj\n", "MLP gate\nproj", "MLP up\nproj",
        "MLP down\nproj"
        ]


        if attackMode == "vis":
            # MODIFIED vs Gemma: Qwen's vision side has 8 tracked points
            # instead of 7 (extra visUp entry), so we drop the last one (the
            # vis-to-lan connector, which is layer-independent) leaving the
            # first 7.
            point_labels = point_labels[:7]
        else:
            # MODIFIED vs Gemma: language items now start at index 8 (was 7).
            point_labels = point_labels[8:]

        PostAttackAlignments = []
        PreAttackAlignments = []
        AlignmnetIncreases = []
        NumSamplesConsidered = 38
        AggregationOverFlattenedAlignmentDistributionsOriginal = []
        AggregationFlattenedAlignmentDistributionsAdversary = []
        for attackSample in range(1,NumSamplesConsidered):
            #attackSample = 1
            # MODIFIED vs Gemma: same dataset directory / filename convention
            # used by qwen/QwenUntargeted_BSA_inference.py, so we load the
            # exact same original images the perturbations were trained
            # against.
            IMAGE_PATH = f"../interpretAttacks/llava_attack/dataSamplesForQuant/{attackSample}.JPEG"


            pil = Image.open(IMAGE_PATH).convert("RGB")
            x_orig01 = pil_to_tensor01(pil).to(device)

            # Build template inputs ONCE (inserts image tokens in input_ids)
            template_inputs = build_template_inputs(processor, QUESTION, pil, device)

            # Clean output: preprocess original (differentiable) then generate
            '''pv_clean, grid_clean = qwen_preprocess_differentiable(x_orig01, processor)

            print("\n=== CLEAN OUTPUT ===")
            clean_text = run_generation_with_pixel_values(model, processor, template_inputs, pv_clean, grid_clean, max_new_tokens=MAX_NEW_TOKENS)
            print(clean_text)'''

            if device.type == "cuda":
                torch.cuda.empty_cache()

            # MODIFIED vs Gemma: load the SAME perturbation naming convention
            # used by qwen/QwenUntargeted_BSA_inference.py (the file that
            # already loads/replays the 50-sample BSA perturbations for
            # Qwen2.5-VL). Gemma's examiner used a much more elaborate
            # "saa_BSAexp"-specific filename (towardsNull/whichMLP/
            # balancingAlpha) that has no Qwen counterpart -- the Qwen BSA
            # trainer (qwen/QwenUntargeted_BSA.py) never produced files with
            # that naming scheme.
            #adv_noise_path = (
            #    f"../interpretAttacks/qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
            #    f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.pt"
            #)

            towardsNull = 0.5
            whichMLP = "gate_proj"
            whichMLPVis = "gate_proj"
            chosenLanLayers = [2]
            chosenVisLayers = [0, 1, 2, 4, 5, 6, 7, 8, 9, 14, 24]
            adv_noise_path = (
                f"../interpretAttacks/qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
                f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
                f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
                f"num_steps_{num_steps}_towardsNull_{towardsNull}_{whichMLP}_{whichMLPVis}_{chosenLanLayers}_{chosenVisLayers}.pt"
            )



            best_delta = torch.load(adv_noise_path, map_location=device).to(device=device, dtype=x_orig01.dtype)

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
                #save_conv_path=conv_path,
                AttackStartLayer = AttackStartLayer,
                numLayerstAtAtime = numLayerstAtAtime,
                allTopRightSingularVectors = allTopRightSingularVectors,
                best_delta = best_delta
            )
            if attackMode == "vis":
                RightSingularInputAlignmentAgainstAdversary = RightSingularInputAlignmentAgainstAdversary[:7]
                FlattenedAlignmentDistributionsAdversary = FlattenedAlignmentDistributionsAdversary[:7]

            else:
                RightSingularInputAlignmentAgainstAdversary = RightSingularInputAlignmentAgainstAdversary[8:]
                FlattenedAlignmentDistributionsAdversary = FlattenedAlignmentDistributionsAdversary[8:]

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
                num_steps=num_steps,
                lr=lr,
                epsilon=epsilon,
                device=device,
                #save_conv_path=conv_path,
                AttackStartLayer = AttackStartLayer,
                numLayerstAtAtime = numLayerstAtAtime,
                allTopRightSingularVectors = allTopRightSingularVectors,
                best_delta = best_delta*0
            )

            if attackMode == "vis":
                RightSingularInputAlignmentAgainstOriginal = RightSingularInputAlignmentAgainstOriginal[:7]
                FlattenedAlignmentDistributionsOriginal = FlattenedAlignmentDistributionsOriginal[:7]

            else:
                RightSingularInputAlignmentAgainstOriginal = RightSingularInputAlignmentAgainstOriginal[8:]
                FlattenedAlignmentDistributionsOriginal = FlattenedAlignmentDistributionsOriginal[8:]


                #print("len(FlattenedAlignmentDistributionsOriginal)", len(FlattenedAlignmentDistributionsOriginal))

            AggregationOverFlattenedAlignmentDistributionsOriginal.append(FlattenedAlignmentDistributionsOriginal)
            AggregationFlattenedAlignmentDistributionsAdversary.append(FlattenedAlignmentDistributionsAdversary)

            '''for i in range(len(FlattenedAlignmentDistributionsOriginal)):
                print("FlattenedAlignmentDistributionsOriginal[i].shape", FlattenedAlignmentDistributionsOriginal[i].shape)'''


            averagedAggregationOverFlattenedAlignmentDistributionsOriginal = [
                torch.stack(elements).mean(dim=0)
                for elements in zip(*AggregationOverFlattenedAlignmentDistributionsOriginal)
            ]

            averagedAggregationFlattenedAlignmentDistributionsAdversary = [
                torch.stack(elements).mean(dim=0)
                for elements in zip(*AggregationFlattenedAlignmentDistributionsAdversary)
            ]


            #print(len(averagedAggregationOverFlattenedAlignmentDistributionsOriginal))
            #for t in averagedAggregationOverFlattenedAlignmentDistributionsOriginal:
                #print("t.shape", t.shape)

            #print(len(averagedAggregationFlattenedAlignmentDistributionsAdversary))
            #for t in averagedAggregationFlattenedAlignmentDistributionsAdversary:
                #print("t_2.shape", t.shape)


            #print(f"done for sample {attackSample}" )
        #print("len(AggregationOverFlattenedAlignmentDistributionsOriginal)", len(AggregationOverFlattenedAlignmentDistributionsOriginal))

        '''for i in range(len(averagedAggregationOverFlattenedAlignmentDistributionsOriginal)):
            #print("FlattenedAlignmentDistributionsAdversary[i].shape", averagedAggregationFlattenedAlignmentDistributionsAdversary[i].shape)
            #print("FlattenedAlignmentDistributionsOriginal[i].shape", averagedAggregationOverFlattenedAlignmentDistributionsOriginal[i].shape)


            print("attackMode", attackMode)
            print("VisionLayerTrack", VisionLayerTrack)
            print("LanLayerTrack", LanLayerTrack)
            orig = averagedAggregationOverFlattenedAlignmentDistributionsOriginal[i].detach().to(torch.float32).cpu().numpy()
            adv = averagedAggregationFlattenedAlignmentDistributionsAdversary[i].detach().to(torch.float32).cpu().numpy()


            l2 = np.linalg.norm(adv - orig)
            mad = np.mean(np.abs(adv - orig))
            max_change = np.max(np.abs(adv - orig))

            print(f"L2 distance          : {l2:.6f}")

            print(f"Mean Abs Difference  : {mad:.6f}")

            print(f"Maximum Change       : {max_change:.6f}")

            com_orig, com_adv, shift = center_of_mass_shift(orig, adv)

            print(f"COM Shift : {shift:.3f}")

            mean_orig, mean_adv, signShift = mean_alignment_shift(orig, adv)
            print(f"Sign Shift : {signShift:.10f}")
            print()'''





        for i in range(len(averagedAggregationOverFlattenedAlignmentDistributionsOriginal)):

            #print("attackMode:", attackMode)
            #print("VisionLayerTrack:", VisionLayerTrack)
            #print("LanLayerTrack:", LanLayerTrack)

            orig = (
                averagedAggregationOverFlattenedAlignmentDistributionsOriginal[i]
                .detach()
                .to(torch.float32)
                .cpu()
                .numpy()
            )

            adv = (
                averagedAggregationFlattenedAlignmentDistributionsAdversary[i]
                .detach()
                .to(torch.float32)
                .cpu()
                .numpy()
            )

            # ---------------- Metrics ---------------- #

            l2 = np.linalg.norm(adv - orig)
            mad = np.mean(np.abs(adv - orig))
            max_change = np.max(np.abs(adv - orig))

            com_orig, com_adv, com_shift = center_of_mass_shift(orig, adv)

            mean_orig, mean_adv, sign_shift = mean_alignment_shift(orig, adv)

            # ---------------- Print ---------------- #

            #print(f"L2 Distance            : {l2:.6f}")
            #print(f"Mean Abs Difference    : {mad:.6f}")
            #print(f"Maximum Change         : {max_change:.6f}")
            #print(f"COM Shift              : {com_shift:.6f}")
            #print(f"Sign Shift             : {sign_shift:.10f}")
            #print()

            # ---------------- Save row ---------------- #

            row = pd.DataFrame([{
                "AttackMode": attackMode,
                "VisionLayerTrack": VisionLayerTrack,
                "LanLayerTrack": LanLayerTrack,
                "HistogramIndex": i,
                "L2 Distance": l2,
                "Mean Absolute Difference": mad,
                "Maximum Change": max_change,
                "COM Shift": com_shift,
                "Sign Shift": sign_shift
            }])

            row.to_csv(
                csv_path,
                mode="a",
                header=write_header,
                index=False
            )

            # Only write the header once
            write_header = False


if __name__ == "__main__":
    main()
