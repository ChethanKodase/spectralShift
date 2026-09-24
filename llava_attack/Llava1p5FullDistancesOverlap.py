

'''

LLaVA-1.5-7b port of qwen/Qwen2p5FullDistancesOverlap.py.

What this script does (identical pipeline to the Qwen version):

  For every image sample 1..38:
    1. Load the original image and the TRAINED ("strong") adversarial
       perturbation delta that was saved for that sample.
    2. Build a "weak" adversary: Gaussian noise scaled by epsilon (same L_inf
       clamp as the strong one).
    3. Run three forward passes (strong adversary, original = delta*0, weak
       adversary). For each tracked linear operator, a forward pre-hook grabs
       the operator's INPUT activations, and we project every (normalized)
       token onto the FULL right-singular-vector basis of that operator's
       weight (per attention head for q/k/v).
    4. Per singular-vector index, compute the RMS-over-tokens L2 distance
         diff = sqrt(mean_tokens((coeffs_orig - coeffs_adv)^2))
       for strong-vs-original and weak-vs-original.
  Then average (and std) over samples and, per tracked operator, save:
    - llava_attack/OverlapDistancesAvg/Bar_<label>_...png              (overlaid bars)
    - llava_attack/OverlapDistancesAvg/weak_vs_strong_l2_summary.csv   (L2 between mean curves)
    - llava_attack/OverlapDistancesAvgStdBands/Bar_<label>_...png      (mean +/- std bands)
    - llava_attack/OverlapDistancesAvgStdBandsNormalized/Bar_<label>_...png (bands, each curve / its own mean)

ARCHITECTURE MAPPING Qwen2.5-VL -> LLaVA-1.5 (from llava_attack/model_parameters.txt):

  Vision tower (CLIP ViT-L/14-336, 24 blocks, hidden 1024, 16 heads):
    Qwen attn.qkv (fused, sliced q/k/v)  -> self_attn.q_proj / k_proj / v_proj (separate Linears)
    Qwen attn.proj                       -> self_attn.out_proj
    Qwen mlp.gate_proj / up_proj / down_proj (SwiGLU) -> mlp.fc1 / mlp.fc2 (2-layer GELU MLP)
  Vision-to-language bridge:
    Qwen visual.merger.mlp[2]            -> multi_modal_projector.linear_2 (4096 -> 4096)
  Language model (LLaMA-7B / Vicuna, 32 layers, hidden 4096, 32 heads, no GQA):
    self_attn.q_proj / k_proj / v_proj, mlp.gate_proj / up_proj / down_proj (same names as Qwen)

  So the tracked-point list is:
    vis: query proj, key proj, value proj, att output proj, MLP fc1, MLP fc2   (6 points, was 7 for Qwen)
    Vis-to-lan proj                                                          (index 6, was 7 for Qwen)
    lan: query, key, value, MLP gate, MLP up, MLP down                         (indices 7.., was 8.. for Qwen)

Model loading follows llava_attack/LlavaUntargted_ChosenSingulaeVectors.py
(LlamaTokenizer + CLIPImageProcessor, no LlavaProcessor, fp16,
"USER: <image>\\n...\\nASSISTANT:" prompt, differentiable resize+center-crop).

Run from the spectralShift/ directory:



export CUDA_VISIBLE_DEVICES=7
conda deactivate
cd spectralShift/
conda activate llava15
export PYTHONNOUSERSITE=1
for StudyLayer in $(seq 0 31); do
    python llava_attack/Llava1p5FullDistancesOverlap.py --attck_type bsa --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --VisionLayerTrack 0 --LanLayerTrack $StudyLayer --kthSingVec -10 --attackMode lan
done



export CUDA_VISIBLE_DEVICES=6
conda deactivate
cd spectralShift/
conda activate llava15
export PYTHONNOUSERSITE=1
for StudyLayer in $(seq 0 23); do
    python llava_attack/Llava1p5FullDistancesOverlap.py --attck_type bsa --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --VisionLayerTrack $StudyLayer --LanLayerTrack 0 --kthSingVec 10 --attackMode vis
done


llava15

export CUDA_VISIBLE_DEVICES=3
cd interpretAttacks/
conda activate llava15

'''


import os
# Must be set before any CUDA/cuBLAS context is created (same as the Qwen script).
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# All outputs are written next to this script, i.e. inside spectralShift/llava_attack,
# regardless of the working directory the script is launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))



# ----------------------------
# Reproducibility (same settings as the Qwen script / LLaVA attack script)
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


def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()


def wasserstein_distance(tensor_a, tensor_b):
    tensor_a_sorted, _ = torch.sort(torch.flatten(tensor_a))
    tensor_b_sorted, _ = torch.sort(torch.flatten(tensor_b))
    return torch.mean(torch.abs(tensor_a_sorted - tensor_b_sorted))


# ----------------------------
# Utilities: image <-> tensor
# ----------------------------
def pil_to_tensor01(pil_img: Image.Image) -> torch.Tensor:
    """PIL RGB -> torch float tensor in [0,1], shape (1,3,H,W)"""
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor01_to_pil(t01: torch.Tensor) -> Image.Image:
    """torch tensor [0,1], shape (1,3,H,W) or (3,H,W) -> PIL RGB"""
    if t01.dim() == 4:
        t01 = t01[0]
    t01 = t01.detach().cpu().clamp(0, 1)
    arr = (t01.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ----------------------------
# LLaVA (CLIP) differentiable preprocessing
# (ported verbatim from llava_attack/LlavaUntargted_ChosenSingulaeVectors.py so the
#  saved perturbations line up with the exact same pipeline they were trained with)
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
# Template inputs (LLaVA-1.5 fixed prompt, raw LlamaTokenizer -- no LlavaProcessor,
# same as llava_attack/LlavaUntargted_ChosenSingulaeVectors.py)
# ----------------------------
def build_template_inputs(tokenizer, question: str, device):
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    enc = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def run_generation_with_pixel_values(model, tokenizer, template_inputs, pixel_values, max_new_tokens=128):
    model.eval()
    inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
    inputs["pixel_values"] = pixel_values
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_only = out_ids[:, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_only[0], skip_special_tokens=True)


# ----------------------------
# Alignment helpers (same math as the Qwen script)
# NOTE: LLaVA runs in fp16 (Qwen used bf16). The projection coefficients and the
# later squared differences are small numbers, so the helpers below upcast to
# float32 before computing them to avoid fp16 underflow / precision loss.
# ----------------------------
def _drop_batch_dim_if_present(t):
    if isinstance(t, (tuple, list)):
        t = t[0]
    if t.dim() == 3 and t.shape[0] == 1:
        return t[0]
    if t.dim() == 3:
        return t.reshape(-1, t.shape[-1])
    return t


def getMeanAlignmentWithTopRightSingularVector(InputToLayer, topRightSingularVector):
    H = _drop_batch_dim_if_present(InputToLayer).to(torch.float32)   # (N, d_in)
    V = topRightSingularVector.to(device=H.device, dtype=torch.float32)  # (d_in, d_in)

    H_hat = F.normalize(H, dim=1)
    V_hat = F.normalize(V, dim=1)

    coeffs = H_hat @ V_hat.T                      # (N, d_in)
    per_token_energy = (coeffs ** 2).sum(dim=1)   # (N,)
    mean_energy = per_token_energy.mean().item()
    # raw (un-reduced) coeffs, as in the Qwen script
    return mean_energy, coeffs


def getMeanAlignmentWithAttentionHeadTopRightSingularVector(InputToLayer, topRightSingularVector):
    H = _drop_batch_dim_if_present(InputToLayer).to(torch.float32)   # (N, d_model)
    V = topRightSingularVector.to(device=H.device, dtype=torch.float32)  # (num_heads, d_model, d_model)

    H_hat = F.normalize(H, dim=1)
    V_hat = F.normalize(V, dim=2)

    coeffs = torch.einsum('nd,hkd->hnk', H_hat, V_hat)  # (num_heads, N, d_model)
    energy = (coeffs ** 2).sum(dim=2)
    mean_energy_all = energy.mean(dim=1).mean().item()
    # raw (un-reduced) coeffs, as in the Qwen script
    return mean_energy_all, coeffs


# ----------------------------------------------------------------------------
# Robust getters for LLaVA's sub-modules. Old transformers (matching
# model_parameters.txt):  model.vision_tower / model.multi_modal_projector /
# model.language_model.model.layers. Newer transformers (>=4.52) wrap them in
# model.model.*, so probe both.
# ----------------------------------------------------------------------------
def get_vision_layers(model):
    for root in (model, getattr(model, "model", None)):
        if root is None:
            continue
        vt = getattr(root, "vision_tower", None)
        if vt is not None and hasattr(vt, "vision_model"):
            return vt.vision_model.encoder.layers
    raise RuntimeError("Could not find LLaVA vision tower encoder layers.")


def get_multi_modal_projector(model):
    for root in (model, getattr(model, "model", None)):
        if root is not None and hasattr(root, "multi_modal_projector"):
            return root.multi_modal_projector
    raise RuntimeError("Could not find LLaVA multi_modal_projector.")


def get_language_layers(model):
    # old: model.language_model.model.layers
    lm = getattr(model, "language_model", None)
    if lm is not None:
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
        if hasattr(lm, "layers"):
            return lm.layers
    # new: model.model.language_model.layers
    inner = getattr(model, "model", None)
    if inner is not None:
        lm = getattr(inner, "language_model", None)
        if lm is not None and hasattr(lm, "layers"):
            return lm.layers
        if hasattr(inner, "layers"):
            return inner.layers
    raise RuntimeError("Could not find LLaVA language-model layers.")


def _resolve_vision_config(model):
    return getattr(model.config, "vision_config", model.config)


def _resolve_text_config(model):
    return getattr(model.config, "text_config", model.config)


# ----------------------------------------------------------------------------
# Hooks (input + output of every tracked operator, same naming as Qwen, with
# FC1/FC2 for CLIP's 2-layer vision MLP instead of Qwen's visGate/visUp/visDown)
# ----------------------------------------------------------------------------
layer_inputs = {}
layer_outputs = {}


def make_pre_hook(name):
    def hook(module, inputs):
        layer_inputs[name] = inputs[0]
    return hook


def make_forward_hook(name):
    def hook(module, inputs, output):
        layer_outputs[name] = output
    return hook


# Order of tracked operators (index in this list == index into
# AlignmentDistributions). Right singular vectors of operator i live at
# allTopRightSingularVectors[2*i], left ones at [2*i + 1] -- same layout as Qwen.
TRACKED_POINTS = [
    # (hook key,   is_head)
    ("qry0",       True),    # 0  vision q_proj
    ("key0",       True),    # 1  vision k_proj
    ("val0",       True),    # 2  vision v_proj
    ("visOutProj", False),   # 3  vision out_proj
    ("visFC1",     False),   # 4  vision mlp.fc1
    ("visFC2",     False),   # 5  vision mlp.fc2
    ("MulModProj", False),   # 6  multi_modal_projector.linear_2
    ("qryLan",     True),    # 7  language q_proj
    ("keyLan",     True),    # 8  language k_proj
    ("valLan",     True),    # 9  language v_proj
    ("gate",       False),   # 10 language mlp.gate_proj
    ("up",         False),   # 11 language mlp.up_proj
    ("down",       False),   # 12 language mlp.down_proj
]
NUM_VIS_POINTS = 6       # Qwen had 7 (gate/up/down); CLIP has fc1/fc2
MERGER_POINT_IDX = 6     # Qwen had 7
LAN_START_IDX = 7        # Qwen had 8


# ----------------------------
# Single forward pass with a given (loaded) perturbation -- post-attack examiner,
# same role as adam_attack_original_space() in the Qwen script.
# ----------------------------
def adam_attack_original_space(
    model,
    image_processor,
    template_inputs,
    x_orig01,               # (1,3,H0,W0) in [0,1]
    attck_type: str,
    num_steps: int,
    lr: float,
    epsilon: float,         # L_inf bound in ORIGINAL pixel space [0,1]
    device,
    AttackStartLayer: int,
    numLayerstAtAtime: int,
    allTopRightSingularVectors,
    best_delta,
):
    x_orig01 = x_orig01.detach().to(device)
    delta = best_delta.detach().to(device=device, dtype=x_orig01.dtype)
    best_delta = delta.clone()

    model.eval()

    # Same sanity check as Qwen (endPos must not exceed #hidden states). The Qwen
    # script ran an extra clean forward only to read len(hidden_states); here it
    # is read from config (num_hidden_layers + 1) to avoid a redundant pass.
    text_cfg = _resolve_text_config(model)
    hiddStateLen = int(getattr(text_cfg, "num_hidden_layers")) + 1
    startPos = AttackStartLayer
    endPos = startPos + numLayerstAtAtime
    if endPos > hiddStateLen:
        raise ValueError(f"endPos ({endPos}) exceeds number of hidden states ({hiddStateLen})")

    adv_inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
    adv_inputs["labels"] = template_inputs["input_ids"]
    adv_inputs["use_cache"] = False

    layer_inputs.clear()
    layer_outputs.clear()

    with torch.no_grad():
        x_adv01 = (x_orig01 + delta).clamp(0.0, 1.0)
        x_adv01 = torch.max(torch.min(x_adv01, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

        pv_adv = llava_preprocess_differentiable(x_adv01, image_processor)
        adv_inputs["pixel_values"] = pv_adv

        outputs = model(**adv_inputs, output_hidden_states=False, return_dict=True)
        del outputs

        RightSingularInputAlignment = []
        AlignmentDistributions = []
        for i, (key, is_head) in enumerate(TRACKED_POINTS):
            InputToLayer = layer_inputs.get(key)
            if InputToLayer is None:
                raise RuntimeError(f"Hook '{key}' did not fire during the forward pass.")
            if is_head:
                mae, ProjDistrib = getMeanAlignmentWithAttentionHeadTopRightSingularVector(
                    InputToLayer, allTopRightSingularVectors[2 * i])
            else:
                mae, ProjDistrib = getMeanAlignmentWithTopRightSingularVector(
                    InputToLayer, allTopRightSingularVectors[2 * i])
            RightSingularInputAlignment.append(mae)
            AlignmentDistributions.append(ProjDistrib)

    layer_inputs.clear()
    layer_outputs.clear()

    RightSingularInputAlignment = np.array(RightSingularInputAlignment)
    # kept un-flattened (raw coeffs, shape (N,k) or (h,N,k)), as in the Qwen script
    FlattenedAlignmentDistributions = list(AlignmentDistributions)

    with torch.no_grad():
        x_adv01_final = (x_orig01 + best_delta).clamp(0.0, 1.0)
        x_adv01_final = torch.max(torch.min(x_adv01_final, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

    return x_adv01_final, best_delta, RightSingularInputAlignment, FlattenedAlignmentDistributions


def _select_mode(lst, attackMode):
    if attackMode == "vis":
        return lst[:NUM_VIS_POINTS]
    return lst[LAN_START_IDX:]


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="LLaVA-1.5 weak-vs-strong adversary singular-vector distance overlap")
    parser.add_argument("--attck_type", type=str, default="bsa",
                        help="attack type tag used in the saved perturbation filename")
    parser.add_argument("--desired_norm_l_inf", type=float, default=0.03,
                        help="epsilon L_inf in ORIGINAL pixel space [0..1]")
    parser.add_argument("--learningRate", type=float, default=1e-3, help="Adam lr (filename tag)")
    parser.add_argument("--num_steps", type=int, default=2000, help="Adam steps (filename tag)")
    parser.add_argument("--AttackStartLayer", type=int, default=0, help="From which layer do you start attack")
    parser.add_argument("--numLayerstAtAtime", type=int, default=2, help="Number of layers taken at a time to attack")
    parser.add_argument("--VisionLayerTrack", type=int, default=2, help="which vision layer to track (0-23)")
    parser.add_argument("--LanLayerTrack", type=int, default=2, help="which language layer to track (0-31)")
    parser.add_argument("--kthSingVec", type=int, default=0, help="kept for compatibility (full basis is always used)")
    parser.add_argument("--attackMode", type=str, default="lan", help="vis or lan")
    parser.add_argument("--advRoot", type=str,
                        default="../interpretAttacks/llava_attack/outputsStorageImagenet/advOutputs",
                        help="root folder holding <attackSample>/adv_ORIG_attackType_..._.pt perturbations")
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
    advRoot = str(args.advRoot)

    MODEL_PATH = "../LLaVA/llava-1.5-7b-hf"
    QUESTION = "What is shown in this image?"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"device={device}, dtype={dtype}")

    # ---- model loading: same as llava_attack/LlavaUntargted_ChosenSingulaeVectors.py ----
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

    # ---- architecture hyper-parameters ----
    vision_layers = get_vision_layers(model)
    language_layers = get_language_layers(model)
    mm_projector = get_multi_modal_projector(model)

    vision_cfg = _resolve_vision_config(model)
    text_cfg = _resolve_text_config(model)

    num_heads = int(getattr(vision_cfg, "num_attention_heads", getattr(vision_cfg, "num_heads", 16)))  # 16
    d_model = int(getattr(vision_cfg, "hidden_size", vision_layers[0].self_attn.q_proj.weight.shape[1]))  # 1024

    d_modelT = int(getattr(text_cfg, "hidden_size"))                          # 4096
    num_headsT = int(getattr(text_cfg, "num_attention_heads"))                # 32
    num_kv_headsT = int(getattr(text_cfg, "num_key_value_heads", num_headsT))  # 32 (no GQA)

    if not (0 <= VisionLayerTrack < len(vision_layers)):
        raise ValueError(f"VisionLayerTrack must be in [0, {len(vision_layers) - 1}]")
    if not (0 <= LanLayerTrack < len(language_layers)):
        raise ValueError(f"LanLayerTrack must be in [0, {len(language_layers) - 1}]")

    print(f"[INFO] vision: {len(vision_layers)} layers, d_model={d_model}, heads={num_heads}")
    print(f"[INFO] language: {len(language_layers)} layers, d_model={d_modelT}, heads={num_headsT}, kv_heads={num_kv_headsT}")
    print(f"[INFO] VisionLayerTrack={VisionLayerTrack} LanLayerTrack={LanLayerTrack} attackMode={attackMode} kthSingVec={kthSingVec}")

    # ---- SVD helpers (full bases; one SVD per operator returns both U and Vh) ----
    def svdLinear(module):
        U, S, Vh = torch.linalg.svd(module.weight.detach().to(torch.float32))
        return Vh, U

    def svdAttentionHeads(module, num_heads_local):
        W = module.weight.detach().to(torch.float32)
        d_out, d_in = W.shape
        d_head_local = d_out // num_heads_local
        W = W.view(num_heads_local, d_head_local, d_in)
        vh_list, u_list = [], []
        for h in range(num_heads_local):
            U, S, Vh = torch.linalg.svd(W[h])
            vh_list.append(Vh)
            u_list.append(U)
        return torch.stack(vh_list, 0), torch.stack(u_list, 0)

    with torch.no_grad():
        vis_block = vision_layers[VisionLayerTrack]
        lan_block = language_layers[LanLayerTrack]

        tracked_modules = {
            "qry0":       (vis_block.self_attn.q_proj, ("heads", num_heads)),
            "key0":       (vis_block.self_attn.k_proj, ("heads", num_heads)),
            "val0":       (vis_block.self_attn.v_proj, ("heads", num_heads)),
            "visOutProj": (vis_block.self_attn.out_proj, ("linear", None)),
            "visFC1":     (vis_block.mlp.fc1, ("linear", None)),
            "visFC2":     (vis_block.mlp.fc2, ("linear", None)),
            "MulModProj": (mm_projector.linear_2, ("linear", None)),
            "qryLan":     (lan_block.self_attn.q_proj, ("heads", num_headsT)),
            "keyLan":     (lan_block.self_attn.k_proj, ("heads", num_kv_headsT)),
            "valLan":     (lan_block.self_attn.v_proj, ("heads", num_kv_headsT)),
            "gate":       (lan_block.mlp.gate_proj, ("linear", None)),
            "up":         (lan_block.mlp.up_proj, ("linear", None)),
            "down":       (lan_block.mlp.down_proj, ("linear", None)),
        }

        hook_handles = []
        for key, (module, _) in tracked_modules.items():
            hook_handles.append(module.register_forward_pre_hook(make_pre_hook(key)))
            hook_handles.append(module.register_forward_hook(make_forward_hook(key)))

        print("[INFO] Computing singular-vector bases ...")
        allTopRightSingularVectors = []
        for key, is_head in TRACKED_POINTS:
            module, (kind, nh) = tracked_modules[key]
            if kind == "heads":
                Vh, U = svdAttentionHeads(module, nh)
            else:
                Vh, U = svdLinear(module)
            allTopRightSingularVectors.append(Vh)   # right singular vectors (used)
            allTopRightSingularVectors.append(U)    # left singular vectors (kept for layout parity)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    point_labels = [
        "query proj", "key proj", "value proj", "att output\nproj",
        "MLP fc1\n(vis)", "MLP fc2\n(vis)", "Vis-to-lan\nproj", "query proj\n",
        "key proj\n", "value proj\n", "MLP gate\nproj", "MLP up\nproj",
        "MLP down\nproj"
    ]
    point_labels = _select_mode(point_labels, attackMode)

    PostAttackAlignments = []
    NumSamplesConsidered = 51
    AggregationOverFlattenedAlignmentDistributionsOriginal = []   # weak vs original
    AggregationFlattenedAlignmentDistributionsAdversary = []      # strong vs original

    for attackSample in range(1, NumSamplesConsidered):
        IMAGE_PATH = f"../interpretAttacks/gemma_attack/dataSamplesForQuant/{attackSample}.JPEG"

        pil = Image.open(IMAGE_PATH).convert("RGB")
        x_orig01 = pil_to_tensor01(pil).to(device)

        template_inputs = build_template_inputs(tokenizer, QUESTION, device)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        '''adv_noise_path = os.path.join(
            advRoot, f"{attackSample}",
            f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.pt"
        )'''


        adv_noise_path = (
            f"../interpretAttacks/llava_attack/outputsStorage/advOutputs/{attackSample}/"
            f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
            f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
            f"num_steps_{num_steps}_.pt"
        )

        #interpretAttacks/llava_attack/outputsStorage/advOutputs/1/adv_ORIG_attackType_bsa_lr_0.001_eps_0.004_AttackStartLayer_0_numLayerstAtAtime_1_num_steps_1000_.pt
        if not os.path.exists(adv_noise_path):
            raise FileNotFoundError(
                f"Adversarial perturbation not found: {adv_noise_path}\n"
                f"Use --advRoot / --attck_type / --learningRate / --desired_norm_l_inf / --num_steps "
                f"to point to the saved LLaVA perturbations."
            )

        best_delta = torch.load(adv_noise_path, map_location=device).to(device=device, dtype=x_orig01.dtype)
        print(f"[sample {attackSample}] best_delta.max()", best_delta.max().item(), "best_delta.min()", best_delta.min().item())

        # weak adversary: gaussian noise, same L_inf bound (clamped inside the pass)
        weak_delta = torch.randn_like(best_delta) * epsilon
        print(f"[sample {attackSample}] weak_delta.max()", weak_delta.max().item(), "weak_delta.min()", weak_delta.min().item())

        common = dict(
            model=model,
            image_processor=image_processor,
            template_inputs=template_inputs,
            x_orig01=x_orig01,
            attck_type=attck_type,
            num_steps=num_steps,
            lr=lr,
            epsilon=epsilon,
            device=device,
            AttackStartLayer=AttackStartLayer,
            numLayerstAtAtime=numLayerstAtAtime,
            allTopRightSingularVectors=allTopRightSingularVectors,
        )

        # strong (trained) adversary
        _, _, RightSingularInputAlignmentAgainstAdversary, FlattenedAlignmentDistributionsAdversary = \
            adam_attack_original_space(best_delta=best_delta, **common)
        RightSingularInputAlignmentAgainstAdversary = _select_mode(RightSingularInputAlignmentAgainstAdversary, attackMode)
        FlattenedAlignmentDistributionsAdversary = _select_mode(FlattenedAlignmentDistributionsAdversary, attackMode)
        PostAttackAlignments.append(RightSingularInputAlignmentAgainstAdversary)

        # original (delta = 0)
        _, _, RightSingularInputAlignmentAgainstOriginal, FlattenedAlignmentDistributionsOriginal = \
            adam_attack_original_space(best_delta=best_delta * 0, **common)
        RightSingularInputAlignmentAgainstOriginal = _select_mode(RightSingularInputAlignmentAgainstOriginal, attackMode)
        FlattenedAlignmentDistributionsOriginal = _select_mode(FlattenedAlignmentDistributionsOriginal, attackMode)

        # weak (gaussian) adversary
        _, _, RightSingularInputAlignmentAgainstWeak, FlattenedAlignmentDistributionsWeak = \
            adam_attack_original_space(best_delta=weak_delta, **common)
        RightSingularInputAlignmentAgainstWeak = _select_mode(RightSingularInputAlignmentAgainstWeak, attackMode)
        FlattenedAlignmentDistributionsWeak = _select_mode(FlattenedAlignmentDistributionsWeak, attackMode)

        # L2 (RMS over tokens) difference vs original, BEFORE token averaging
        DiffStrongThisSample = []
        DiffWeakThisSample = []
        for i in range(len(FlattenedAlignmentDistributionsOriginal)):
            orig_c = FlattenedAlignmentDistributionsOriginal[i]
            adv_c = FlattenedAlignmentDistributionsAdversary[i]
            weak_c = FlattenedAlignmentDistributionsWeak[i]

            token_dim = 0 if orig_c.dim() == 2 else 1   # (N,k) -> 0 ; (h,N,k) -> 1

            diffStrong = (orig_c - adv_c).pow(2).mean(dim=token_dim).sqrt().flatten()
            diffWeak = (orig_c - weak_c).pow(2).mean(dim=token_dim).sqrt().flatten()

            # keep only the reduced vectors on CPU across samples
            DiffStrongThisSample.append(diffStrong.detach().cpu())
            DiffWeakThisSample.append(diffWeak.detach().cpu())

        AggregationOverFlattenedAlignmentDistributionsOriginal.append(DiffWeakThisSample)
        AggregationFlattenedAlignmentDistributionsAdversary.append(DiffStrongThisSample)

        del FlattenedAlignmentDistributionsAdversary, FlattenedAlignmentDistributionsOriginal, FlattenedAlignmentDistributionsWeak
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- aggregate over samples ----
    averagedAggregationOverFlattenedAlignmentDistributionsOriginal = [
        torch.stack(elements).mean(dim=0) for elements in zip(*AggregationOverFlattenedAlignmentDistributionsOriginal)
    ]
    averagedAggregationFlattenedAlignmentDistributionsAdversary = [
        torch.stack(elements).mean(dim=0) for elements in zip(*AggregationFlattenedAlignmentDistributionsAdversary)
    ]
    averagedAggregationOverFlattenedAlignmentDistributionsOriginalSTD = [
        torch.stack(elements).std(dim=0) for elements in zip(*AggregationOverFlattenedAlignmentDistributionsOriginal)
    ]
    averagedAggregationFlattenedAlignmentDistributionsAdversarySTD = [
        torch.stack(elements).std(dim=0) for elements in zip(*AggregationFlattenedAlignmentDistributionsAdversary)
    ]

    save_dir = os.path.join(SCRIPT_DIR, "OverlapDistancesAvg")
    save_dirBandsRaw = os.path.join(SCRIPT_DIR, "OverlapDistancesAvgStdBands")
    save_dirBandsNorm = os.path.join(SCRIPT_DIR, "OverlapDistancesAvgStdBandsNormalized")
    for d in (save_dir, save_dirBandsRaw, save_dirBandsNorm):
        os.makedirs(d, exist_ok=True)

    for i in range(len(averagedAggregationOverFlattenedAlignmentDistributionsOriginal)):
        weak = averagedAggregationOverFlattenedAlignmentDistributionsOriginal[i].to(torch.float32).numpy()
        strong = averagedAggregationFlattenedAlignmentDistributionsAdversary[i].to(torch.float32).numpy()
        print(f"[{i}] strong.shape {strong.shape}  weak.shape {weak.shape}")

        weak_mean = np.mean(weak)
        strong_mean = np.mean(strong)
        weak_norm = weak / weak_mean
        strong_norm = strong / strong_mean

        L = len(weak)
        x = np.arange(L)

        label = point_labels[i].replace("\n", " ")
        fname = (
            f"Bar_{label.replace(' ', '_')}_attackSample_{attackSample}_attackMode_{attackMode}"
            f"_LanLayerTrack_{LanLayerTrack})_VisionLayerTrack_{VisionLayerTrack}.png"
        )

        # ---------------- 1) overlaid bars ----------------
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.bar(x, strong, width=1.0, color="red", edgecolor="none", alpha=0.6,
               label="Strong Adversary (Trained) vs Original", zorder=2)
        ax.bar(x, weak, width=1.0, color="blue", edgecolor="none", alpha=0.6,
               label="Weak Adversary (Gaussian) vs Original", zorder=1)
        ax.set_title(f"{label} — L2 Distance: Weak vs Strong Adversary")
        ax.set_xlabel("<- top singular vector   |   bottom singular vector ->")
        ax.set_ylabel("L2 Distance")
        ax.legend()
        plt.tight_layout()
        save_path = os.path.join(save_dir, fname)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")

        weak_vs_strong_l2 = float(np.linalg.norm(strong - weak))
        csv_path = os.path.join(save_dir, "weak_vs_strong_l2_summary.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as csv_f:
            csv_writer = csv.writer(csv_f)
            if write_header:
                csv_writer.writerow(["LanLayerTrack", "VisionLayerTrack", "attackMode",
                                     "point_label", "weak_vs_strong_l2_distance"])
            csv_writer.writerow([LanLayerTrack, VisionLayerTrack, attackMode, label, weak_vs_strong_l2])
        print(f"Logged weak-vs-strong L2 distance ({weak_vs_strong_l2:.6f}) to: {csv_path}")

        weak_std = averagedAggregationOverFlattenedAlignmentDistributionsOriginalSTD[i].to(torch.float32).numpy()
        strong_std = averagedAggregationFlattenedAlignmentDistributionsAdversarySTD[i].to(torch.float32).numpy()
        weak_std_norm = weak_std / weak_mean
        strong_std_norm = strong_std / strong_mean

        # ---------------- 2) mean +/- std bands ----------------
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.fill_between(x, weak - weak_std, weak + weak_std, color="blue", alpha=0.35, linewidth=0, zorder=2.1)
        ax.plot(x, weak, color="blue", linewidth=0.6, zorder=2.2, label="Weak Adversary (Gaussian) vs Original")
        ax.fill_between(x, strong - strong_std, strong + strong_std, color="red", alpha=0.35, linewidth=0, zorder=2.3)
        ax.plot(x, strong, color="red", linewidth=0.6, zorder=2.4, label="Strong Adversary (Trained) vs Original")
        ax.set_title(f"{label} — L2 Distance: Weak vs Strong Adversary (Mean ± STD)")
        ax.set_xlabel("<- top singular vector   |   bottom singular vector ->")
        ax.set_ylabel("L2 Distance")
        ax.legend()
        plt.tight_layout()
        save_pathBands = os.path.join(save_dirBandsRaw, fname)
        plt.savefig(save_pathBands, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_pathBands}")

        # ---------------- 3) normalized mean +/- std bands ----------------
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.fill_between(x, weak_norm - weak_std_norm, weak_norm + weak_std_norm, color="blue", alpha=0.2, linewidth=0, zorder=2.1)
        ax.plot(x, weak_norm, color="blue", linewidth=0.6, zorder=2.2, label="Weak Adversary (Gaussian) vs Original")
        ax.fill_between(x, strong_norm - strong_std_norm, strong_norm + strong_std_norm, color="red", alpha=0.2, linewidth=0, zorder=2.3)
        ax.plot(x, strong_norm, color="red", linewidth=0.6, zorder=2.4, label="Strong Adversary (Trained) vs Original")
        ax.set_title(f"{label} — L2 Distance: Weak vs Strong Adversary (Mean ± STD)")
        ax.set_xlabel("<- top singular vector   |   bottom singular vector ->")
        ax.set_ylabel("L2 Distance")
        ax.legend()
        plt.tight_layout()
        save_pathBands = os.path.join(save_dirBandsNorm, fname)
        plt.savefig(save_pathBands, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_pathBands}")

    for h in hook_handles:
        h.remove()


if __name__ == "__main__":
    main()
