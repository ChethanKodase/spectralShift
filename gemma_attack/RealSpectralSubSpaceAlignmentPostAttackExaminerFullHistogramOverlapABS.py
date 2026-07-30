



'''


---

export CUDA_VISIBLE_DEVICES=5
conda activate gemma3
cd spectralShift
for StudyLayer in $(seq 0 33); do
    python gemma_attack/RealSpectralSubSpaceAlignmentPostAttackExaminerFullHistogramOverlapABS.py --attck_type saa_BSAexp --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --VisionLayerTrack 0 --LanLayerTrack $StudyLayer --kthSingVec -10 --attackMode lan
done




export CUDA_VISIBLE_DEVICES=4
conda activate gemma3
cd spectralShift
for StudyLayer in $(seq 0 26); do
    python gemma_attack/RealSpectralSubSpaceAlignmentPostAttackExaminerFullHistogramOverlapABS.py --attck_type saa_BSAexp --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --VisionLayerTrack $StudyLayer --LanLayerTrack 0 --kthSingVec 10 --attackMode vis
done




'''



import os
import sys
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
# Differentiable preprocessing (approx Gemma)
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


def getMeanAlignmentWithTopRightSingularVector(InputToLayer, topRightSingularVector):


    H = InputToLayer[0]        
    V = topRightSingularVector   
    V = V.to(H)

    H_hat = F.normalize(H, dim=1)      
    V_hat = F.normalize(V, dim=1)     

    coeffs = H_hat @ V_hat.T        # (N, k)

    # Energy per token (sum of squared coefficients)
    per_token_energy = (coeffs ** 2).sum(dim=1)  # (N,)

    #print("coeffs.shape", coeffs.shape)
    #print("abs(coeffs).mean(dim=1).shape", abs(coeffs).mean(dim=1).shape)

    #coeffHIst = abs(coeffs).mean(dim=1)

    # Mean across tokens
    mean_energy = per_token_energy.mean().item()

    return mean_energy, abs(coeffs).mean(dim=1)

def getMeanAlignmentWithTopLeftSingularVector(InputToLayer, topRightSingularVector):
    v = topRightSingularVector.to(InputToLayer)
    v_hat = v / v.norm()
    h_hat = InputToLayer[0] / InputToLayer[0].norm(dim=-1, keepdim=True)
    dots = h_hat @ v_hat
    dots = dots.squeeze(0)
    mean_abs_value = dots.abs().mean().item()
    return mean_abs_value


def getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, topRightSingularVector):
    #print("InputToLayer.shape", InputToLayer.shape)
    H = InputToLayer[0]          # (4096, 1152)
    V = topRightSingularVector   # (16, 1152)
    V = V.to(H)
    #print("H.shape", H.shape)
    #print("V.shape", V.shape)
    H_hat = F.normalize(H, dim=1)        # (4096, 1152)
    V_hat = F.normalize(V, dim=2)        # (16, 2, 1152)


    #print("H_hat.shape", H_hat.shape)
    #print("V_hat.shape", V_hat.shape)

    coeffs = torch.einsum('nd,hkd->hnk', H_hat, V_hat) # this gave me the dot product between "all the bottom k singular vectors from all the heads on the weight matrix", and all the input tokens. Now I have so many dot product values 16 heads x 4096 tokens number of dot prducts executed and values stored. 

    #print("coeffs.shape", coeffs.shape)
    #now I will perform l2 norm of the dot products corresponding to all the nottom k singular vectors. That is nothing but squaring those singular vectors and adding. This equivalent to l2 norm , squaring,  adding, square rooting and then again squaring
    energy = (coeffs ** 2).sum(dim=2) # right now I have per-token, per-head subspace energy.
    mean_energy_per_head = energy.mean(dim=1)  # (16,)
    mean_energy_all = mean_energy_per_head.mean().item()
    #dots = H_hat @ V_hat.T               # (4096, 16)
    #mean_abs_value = dots.abs().mean().item()

    return mean_energy_all, abs(coeffs).mean(dim=1)


'''



def getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, topRightSingularVector):

    H = InputToLayer[0]          # (4096, 1152)
    V = topRightSingularVector   # (16, 1152)
    V = V.to(H)
    H_hat = F.normalize(H, dim=1)        # (4096, 1152)
    V_hat = F.normalize(V, dim=1)        # (16, 1152)

    dots = H_hat @ V_hat.T               # (4096, 16)
    mean_abs_value = dots.abs().mean().item()

    return mean_abs_value


'''


def getMeanAlignmentWithAttentionHeadTopLeftSingularVector(OutputOfLayer, num_headsT, topLeftSingularVector):

    H = OutputOfLayer[0]          # (4096, 1152)
    V = topLeftSingularVector   # (16, 1152)
    V = V.to(H)
    H_hat = F.normalize(H, dim=1)        # (4096, 1152)
    V_hat = F.normalize(V, dim=1)        # (16, 1152)
    #print()

    H_hat = H_hat.view(H_hat.shape[0], num_headsT, -1)
    #print("after")
    #print("H_hat.shape", H_hat.shape)
    #print("V_hat.shape", V_hat.shape)
    dots = H_hat @ V_hat.T               # (4096, 16)
    mean_abs_value = dots.abs().mean().item()

    return mean_abs_value


def getMeanAlignmentWithLanAttentionHeadTopRightSingularVector(InputToLayer, topRightSingularVector):
    #print("InputToLayer.shape", InputToLayer.shape)
    H = InputToLayer[0]          # (4096, 1152)
    V = topRightSingularVector   # (16, 1152)
    V = V.to(H)
    H_hat = F.normalize(H, dim=1)        # (4096, 1152)
    V_hat = F.normalize(V, dim=1)        # (16, 1152)
    #print("V_hat.shape", V_hat.shape)
    #print("H_hat.shape", H_hat.shape)
    dots = H_hat @ V_hat.T               # (4096, 16)
    mean_abs_value = dots.abs().mean().item()

    return mean_abs_value






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
# ORIGINAL-SPACE Adam attack
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
        layer0 = model.vision_tower.vision_model.encoder.layers[0]
        num_heads = layer0.self_attn.num_heads
        for name, param in model.vision_tower.vision_model.embeddings.named_parameters():
            #print(f"{name:60s} {tuple(param.shape)}")
            if len(param.shape)==4:
                d_model = param.shape[0]
                d_head = d_model // num_heads
                break
        #----------------------attention head hyper parameters extraction------------------------

        d_modelT = model.language_model.config.hidden_size
        num_headsT = model.language_model.config.num_attention_heads


    adv_inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
    adv_inputs["labels"] = template_inputs["input_ids"]
    adv_inputs["use_cache"] = False
    RightSingularInputAlignmentWhole = []
    LeftSingularOutputStepAlignmentWhole = []
    #for step in range(1):



    x_adv01 = (x_orig01 + delta).clamp(0.0, 1.0)
    x_adv01 = torch.max(torch.min(x_adv01, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

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

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        InputToLayer = key0_inputs.get("key0_in")
        key0_in_MAE, ProjDistrib = getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, allTopRightSingularVectors[2])
        #OutputOfLayer = key0_outputs.get("key0_out")
        #key0_out_MAE = getMeanAlignmentWithAttentionHeadTopLeftSingularVector(OutputOfLayer, num_heads, allTopRightSingularVectors[3])
        #print("key0_in_MAE, key0_out_MAE", key0_in_MAE, key0_out_MAE)
        RightSingularInputAlignment.append(key0_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        InputToLayer = val0_inputs.get("val0_in")
        val0_in_MAE, ProjDistrib = getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, allTopRightSingularVectors[4])
        #OutputOfLayer = val0_outputs.get("val0_out")
        #val0_out_MAE = getMeanAlignmentWithAttentionHeadTopLeftSingularVector(OutputOfLayer, num_heads, allTopRightSingularVectors[5])
        #print("val0_in_MAE, val0_out_MAE", val0_in_MAE, val0_out_MAE)
        RightSingularInputAlignment.append(val0_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        InputToLayer = visOutProj_inputs.get("visOutProj_in")
        visOutProj_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[6])
        #OutputOfLayer = visOutProj_outputs.get("visOutProj_out")
        #visOutProj_out_MAE = getMeanAlignmentWithTopLeftSingularVector(OutputOfLayer, allTopRightSingularVectors[7])
        #print("visOutProj_in_MAE, visOutProj_out_MAE", visOutProj_in_MAE ,visOutProj_out_MAE)
        RightSingularInputAlignment.append(visOutProj_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        InputToLayer = FC1_inputs.get("FC1_in")
        FC1_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[8])
        #OutputOfLayer = FC1_outputs.get("FC1_out")
        #FC1_out_MAE = getMeanAlignmentWithTopLeftSingularVector(OutputOfLayer, allTopRightSingularVectors[9])
        #print("FC1_in_MAE, FC1_out_MAE", FC1_in_MAE, FC1_out_MAE)
        RightSingularInputAlignment.append(FC1_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        InputToLayer = FC2_inputs.get("FC2_in")
        FC2_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[10])
        #OutputOfLayer = FC2_outputs.get("FC2_out")
        #FC2_out_MAE = getMeanAlignmentWithTopLeftSingularVector(OutputOfLayer, allTopRightSingularVectors[11])
        #print("FC2_in_MAE, FC2_out_MAE", FC2_in_MAE, FC2_out_MAE)
        RightSingularInputAlignment.append(FC2_in_MAE)
        AlignmentDistributions.append(ProjDistrib)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        InputToLayer = MulModProj_inputs.get("MulModProj_in")
        MulModProj_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[12])
        #OutputOfLayer = MulModProj_outputs.get("MulModProj_out")
        #MulModProj_out_MAE = getMeanAlignmentWithTopLeftSingularVector(OutputOfLayer, allTopRightSingularVectors[13])
        #print("MulModProj_in_MAE MulModProj_out_MAE", MulModProj_in_MAE, MulModProj_out_MAE)
        RightSingularInputAlignment.append(MulModProj_in_MAE)
        AlignmentDistributions.append(ProjDistrib)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        InputToLayer = qryLan_inputs.get("qryLan_in")
        qryLan_in_MAE, ProjDistrib = getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, allTopRightSingularVectors[14])
        #OutputOfLayer = qryLan_outputs.get("qryLan_out")
        #qryLan_out_MAE = getMeanAlignmentWithAttentionHeadTopLeftSingularVector(OutputOfLayer, num_headsT, allTopRightSingularVectors[15])
        #print("qryLan_in_MAE qryLan_out_MAE", qryLan_in_MAE, qryLan_out_MAE)
        RightSingularInputAlignment.append(qryLan_in_MAE)
        AlignmentDistributions.append(ProjDistrib)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        InputToLayer = keyLan_inputs.get("keyLan_in")
        keyLan_in_MAE, ProjDistrib = getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, allTopRightSingularVectors[16])
        #OutputOfLayer = keyLan_outputs.get("keyLan_out")
        #keyLan_out_MAE = getMeanAlignmentWithAttentionHeadTopLeftSingularVector(OutputOfLayer, num_headsT, allTopRightSingularVectors[17])
        #print("keyLan_in_MAE keyLan_out_MAE", keyLan_in_MAE, keyLan_out_MAE)
        RightSingularInputAlignment.append(keyLan_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        InputToLayer = valLan_inputs.get("valLan_in")
        valLan_in_MAE, ProjDistrib = getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, allTopRightSingularVectors[18])
        #OutputOfLayer = valLan_outputs.get("valLan_out")
        #valLan_out_MAE = getMeanAlignmentWithAttentionHeadTopLeftSingularVector(OutputOfLayer, num_headsT, allTopRightSingularVectors[19])
        #print(" valLan_in_MAE,  valLan_out_MAE ", valLan_in_MAE,  valLan_out_MAE)
        RightSingularInputAlignment.append(valLan_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        InputToLayer = GateLayerInputs.get("gate_in")
        gate_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[20])
        #OutputOfLayer = GateLayerOutputs.get("gate_out")
        #gate_out_MAE = getMeanAlignmentWithTopLeftSingularVector(OutputOfLayer, allTopRightSingularVectors[21])
        #print("gate_in_MAE, gate_out_MAE", gate_in_MAE, gate_out_MAE)
        RightSingularInputAlignment.append(gate_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        InputToLayer = UpLayerInputs.get("up_in")
        up_in_MAE, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(InputToLayer, allTopRightSingularVectors[22])
        #OutputOfLayer = UpLayerOutputs.get("up_out")
        #up_out_MAE = getMeanAlignmentWithTopLeftSingularVector(OutputOfLayer, allTopRightSingularVectors[23])
        #print("up_out, up_out_MAE", up_in_MAE, up_out_MAE)
        RightSingularInputAlignment.append(up_in_MAE)
        AlignmentDistributions.append(ProjDistrib)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
    parser = argparse.ArgumentParser(description="Gemma-3 ORIGINAL-image-space adversarial attack (no squeeze)")
    parser.add_argument("--attck_type", type=str, default="grill_l2",
                        help="grill_l2 | grill_cos | OA_l2 | OA_cos")
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


    MODEL_PATH = "../illcond/gemma_attack/Gemma3-4b"
    QUESTION = "What is shown in this image?"
    MAX_NEW_TOKENS = 128


    #attackMode = "lan"
    #attackMode = "vis"

    #IMAGE_PATH = f"gemma_attack/dataSamples/interference68.jpeg"
    

    #os.makedirs("gemma_attack/outputsStorageImagenet", exist_ok=True)
    #os.makedirs(f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}", exist_ok=True)
    #os.makedirs(f"gemma_attack/outputsStorageImagenet/convergence/{attackSample}", exist_ok=True)

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
    layer0 = model.vision_tower.vision_model.encoder.layers[0]
    num_heads = layer0.self_attn.num_heads
    for name, param in model.vision_tower.vision_model.embeddings.named_parameters():
        #print(f"{name:60s} {tuple(param.shape)}")
        if len(param.shape)==4:
            d_model = param.shape[0]
            d_head = d_model // num_heads
            break
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

    d_modelT = model.language_model.config.hidden_size
    num_headsT = model.language_model.config.num_attention_heads

    #print("num_headsT", num_headsT)
    def getTopRightSingularVectorForLanAttentioHeads(qryParam):
        param = qryParam.weight.to(torch.float32)
        d_headT = param.shape[0] // num_headsT
        param_heads = param.view(num_headsT, d_headT, d_modelT)
        query_vh_per_head = []
        for h in range(num_headsT):
            Wh = param_heads[h]      
            U, S, Vh = torch.linalg.svd(Wh.to(torch.float32) )
            if kthSingVec<0:
                query_vh_per_head.append(Vh[:])
            else:
                query_vh_per_head.append(Vh[:])
        query_vh_per_head = torch.stack(query_vh_per_head, 0)
        return query_vh_per_head

    def getTopLeftSingularVectorForLanAttentioHeads(qryParam):
        param = qryParam.weight.to(torch.float32)
        d_headT = param.shape[0] // num_headsT
        param_heads = param.view(num_headsT, d_headT, d_modelT)
        query_vh_per_head = []
        for h in range(num_headsT):
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
        qryParam = model.vision_tower.vision_model.encoder.layers[VisionLayerTrack].self_attn.q_proj
        hook_handle = qryParam.register_forward_pre_hook(qry0_pre_hook)
        hook_handle = qryParam.register_forward_hook(qry0_forward_hook)
        #------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------
        keyParam = model.vision_tower.vision_model.encoder.layers[VisionLayerTrack].self_attn.k_proj
        hook_handle = keyParam.register_forward_pre_hook(key0_pre_hook)
        hook_handle = keyParam.register_forward_hook(key0_forward_hook)
        #------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------
        valParam = model.vision_tower.vision_model.encoder.layers[VisionLayerTrack].self_attn.v_proj
        hook_handle = valParam.register_forward_pre_hook(val0_pre_hook)
        hook_handle = valParam.register_forward_hook(val0_forward_hook)
        #------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------

        visOutProjParam = model.vision_tower.vision_model.encoder.layers[VisionLayerTrack].self_attn.out_proj
        hook_handle = visOutProjParam.register_forward_pre_hook(visOutProj_pre_hook)
        hook_handle = visOutProjParam.register_forward_hook(visOutProj_forward_hook)

        #------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------

        FC1Param = model.vision_tower.vision_model.encoder.layers[VisionLayerTrack].mlp.fc1
        hook_handle = FC1Param.register_forward_pre_hook(FC1_pre_hook)
        hook_handle = FC1Param.register_forward_hook(FC1_forward_hook)
        #------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------

        FC2Param = model.vision_tower.vision_model.encoder.layers[VisionLayerTrack].mlp.fc2
        hook_handle = FC2Param.register_forward_pre_hook(FC2_pre_hook)
        hook_handle = FC2Param.register_forward_hook(FC2_forward_hook)

        #------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------

        #------------multimodal projectrion hook --------------
        MulModProjParam = model.multi_modal_projector
        hook_handle = MulModProjParam.register_forward_pre_hook(MulModProj_pre_hook)
        hook_handle = MulModProjParam.register_forward_hook(MulModProj_forward_hook)
        # ------- language hooks begin ----------------

        qryLanParam = model.language_model.model.layers[LanLayerTrack].self_attn.q_proj
        hook_handle = qryLanParam.register_forward_pre_hook(qryLan_pre_hook)
        hook_handle = qryLanParam.register_forward_hook(qryLan_forward_hook)


        keyLanParam = model.language_model.model.layers[LanLayerTrack].self_attn.k_proj
        hook_handle = keyLanParam.register_forward_pre_hook(keyLan_pre_hook)
        hook_handle = keyLanParam.register_forward_hook(keyLan_forward_hook)

        valLanParam = model.language_model.model.layers[LanLayerTrack].self_attn.v_proj
        hook_handle = valLanParam.register_forward_pre_hook(valLan_pre_hook)
        hook_handle = valLanParam.register_forward_hook(valLan_forward_hook)


        gate_proj = model.language_model.model.layers[LanLayerTrack].mlp.gate_proj # layer 0 doing great
        hook_handle = gate_proj.register_forward_pre_hook(gate_proj_pre_hook)
        hook_handle = gate_proj.register_forward_hook(gate_proj_forward_hook)

        up_proj = model.language_model.model.layers[LanLayerTrack].mlp.up_proj # layer 0 doing great
        hook_handle = up_proj.register_forward_pre_hook(up_proj_pre_hook)
        hook_handle = up_proj.register_forward_hook(up_proj_forward_hook)


        down_proj = model.language_model.model.layers[LanLayerTrack].mlp.down_proj # layer 0 doing great
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

                                      getTopRightSingularVectorForLanAttentioHeads(qryLanParam),
                                      getTopLeftSingularVectorForLanAttentioHeads(qryLanParam),

                                      getTopRightSingularVectorForLanAttentioHeads(keyLanParam),
                                      getTopLeftSingularVectorForLanAttentioHeads(keyLanParam),

                                      getTopRightSingularVectorForLanAttentioHeads(valLanParam),
                                      getTopLeftSingularVectorForLanAttentioHeads(valLanParam),

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
    "MLP exp\n(vis)", "MLP out\nproj (vis)", "Vis-to-lan\nproj", "query proj\n(lan)",
    "key proj\n(lan)", "value proj\n(lan)", "MLP gate\nproj(lan)", "MLP up\nproj (lan)",
    "MLP down\nproj (lan)"
    ]'''

    point_labels = [
    "query proj", "key proj", "value proj", "att output\nproj",
    "MLP exp", "MLP out\nproj", "Vis-to-lan\nproj", "query proj\n",
    "key proj\n", "value proj\n", "MLP gate\nproj", "MLP up\nproj",
    "MLP down\nproj"
    ]


    if attackMode == "vis":
        #point_labels = point_labels[:7]
        point_labels = point_labels[:6]
    else:
        #point_labels = point_labels[6:]
        point_labels = point_labels[7:]

    PostAttackAlignments = []
    PreAttackAlignments = []
    AlignmnetIncreases = []
    NumSamplesConsidered = 50
    AggregationOverFlattenedAlignmentDistributionsOriginal = []
    AggregationFlattenedAlignmentDistributionsAdversary = []
    for attackSample in range(1,NumSamplesConsidered):
        #attackSample = 1
        #IMAGE_PATH = f"gemma_attack/dataSamplesForQuant/{attackSample}.JPEG"
        IMAGE_PATH = f"../interpretAttacks/gemma_attack/dataSamplesForQuant/{attackSample}.JPEG"


        pil = Image.open(IMAGE_PATH).convert("RGB")
        x_orig01 = pil_to_tensor01(pil).to(device)

        # Build template inputs ONCE (inserts image tokens in input_ids)
        template_inputs = build_template_inputs(processor, QUESTION, pil, device)

        # Clean output: preprocess original (differentiable) then generate
        '''pv_clean = gemma_preprocess_differentiable(x_orig01, processor)

        print("\n=== CLEAN OUTPUT ===")
        clean_text = run_generation_with_pixel_values(model, processor, template_inputs, pv_clean, max_new_tokens=MAX_NEW_TOKENS)
        print(clean_text)'''

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Attack
        #conv_path = f"gemma_attack/outputsStorageImagenet/convergence/{attackSample}/gemma_ORIG_attack_{attck_type}_lr_{lr}_eps_{epsilon}_AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_num_steps_{num_steps}_.npy"

        '''if attackMode == "lan":
            adv_noise_path = f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_num_steps_{num_steps}_.pt"
        else:
            adv_noise_path = f"gemma_attack/outputsStorageImagenet/advOutputsVis/{attackSample}/adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_num_steps_{num_steps}_.pt"'''

        #adv_noise_path = f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.pt"

        towardsNullR = 0.15
        AttackStartLayerR = 0
        numLayerstAtAtimeR = 2
        whichMLPR = "up_proj"
        whichMLPvisR = "fc2"
        balancingAlphaR = 0.5
        '''adv_noise_path = (
            f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
            f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
            f"AttackStartLayer_{AttackStartLayerR}_numLayerstAtAtime_{numLayerstAtAtimeR}_"
            f"num_steps_{num_steps}_towardsNull_{towardsNullR}_"
            f"lanMLP_{whichMLPR}_visMLP_{whichMLPvisR}_"
            f"lanLayers_upto4_visLayers_all_balancingAlpha_{balancingAlphaR}.txt"
        )'''

        adv_noise_path = (
            f"../interpretAttacks/gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
            f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
            f"AttackStartLayer_{AttackStartLayerR}_numLayerstAtAtime_{numLayerstAtAtimeR}_"
            f"num_steps_{num_steps}_towardsNull_{towardsNullR}_"
            f"lanMLP_{whichMLPR}_visMLP_{whichMLPvisR}_"
            f"lanLayers_upto4_visLayers_all_balancingAlpha_{balancingAlphaR}.pt"
        )

        best_delta = torch.load(adv_noise_path).to(device)

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
            #RightSingularInputAlignmentAgainstAdversary = RightSingularInputAlignmentAgainstAdversary[:7]
            RightSingularInputAlignmentAgainstAdversary = RightSingularInputAlignmentAgainstAdversary[:6]
            FlattenedAlignmentDistributionsAdversary = FlattenedAlignmentDistributionsAdversary[:6]

        else:
            #RightSingularInputAlignmentAgainstAdversary = RightSingularInputAlignmentAgainstAdversary[6:]
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
            #RightSingularInputAlignmentAgainstOriginal = RightSingularInputAlignmentAgainstOriginal[:7]
            RightSingularInputAlignmentAgainstOriginal = RightSingularInputAlignmentAgainstOriginal[:6]
            FlattenedAlignmentDistributionsOriginal = FlattenedAlignmentDistributionsOriginal[:6]

        else:
            #RightSingularInputAlignmentAgainstOriginal = RightSingularInputAlignmentAgainstOriginal[6:]
            RightSingularInputAlignmentAgainstOriginal = RightSingularInputAlignmentAgainstOriginal[7:]
            FlattenedAlignmentDistributionsOriginal = FlattenedAlignmentDistributionsOriginal[7:]


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


        #print(len(averagedAggregationOverFlattenedAlignmentDistributionsOriginal))          # 6
        #for t in averagedAggregationOverFlattenedAlignmentDistributionsOriginal:
            #print("t.shape", t.shape)             # 20480, 20480, 20480, 276, 276, 276

        #print(len(averagedAggregationFlattenedAlignmentDistributionsAdversary))          # 6
        #for t in averagedAggregationFlattenedAlignmentDistributionsAdversary:
            #print("t_2.shape", t.shape)             # 20480, 20480, 20480, 276, 276, 276


        #print(f"done for sample {attackSample}" )
    #print("len(AggregationOverFlattenedAlignmentDistributionsOriginal)", len(AggregationOverFlattenedAlignmentDistributionsOriginal))

    for i in range(len(averagedAggregationOverFlattenedAlignmentDistributionsOriginal)):
            print("FlattenedAlignmentDistributionsAdversary[i].shape", averagedAggregationFlattenedAlignmentDistributionsAdversary[i].shape)
            print("FlattenedAlignmentDistributionsOriginal[i].shape", averagedAggregationOverFlattenedAlignmentDistributionsOriginal[i].shape)

            orig = averagedAggregationOverFlattenedAlignmentDistributionsOriginal[i].detach().to(torch.float32).cpu().numpy()
            adv = averagedAggregationFlattenedAlignmentDistributionsAdversary[i].detach().to(torch.float32).cpu().numpy()

            orig = (orig)
            adv = (adv)

            L = len(orig)
            x = np.arange(L)

            label = point_labels[i].replace("\n", " ")

            fig, ax = plt.subplots(figsize=(10, 3.5))

            # Adversarial drawn first (background), Original drawn second so it overlaps on top.
            ax.bar(x, adv, width=1.0, color="red", edgecolor="none", alpha=0.6, label="Attacked", zorder=1)
            ax.bar(x, orig, width=1.0, color="green", edgecolor="none", alpha=0.6, label="Original", zorder=2)

            ax.set_title(f"{label} — Original over Attacked")
            ax.set_xlabel("<- top singular vector   |   bottom singular vector ->")
            ax.set_ylabel("Alignment")
            ax.legend()

            plt.tight_layout()

            save_dir = f"gemma_attack/OverlapHistoGramsAvgABS"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(
                save_dir,
                f"Bar_{label.replace(' ', '_')}_attackSample_{attackSample}_attackMode_{attackMode}_LanLayerTrack_{LanLayerTrack})_VisionLayerTrack_{VisionLayerTrack}.png"
            )
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()
            print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()

