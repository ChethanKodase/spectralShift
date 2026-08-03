


'''

Full-coverage variant of QwenUntargted_spectSSGRA.py.

Instead of restricting the attack to a single MLP projection type
(e.g. only "gate_proj") on a hand-picked subset of layers, this script
attacks EVERY linear-projection layer type in EVERY block of the
vision tower, the vision-to-language merger, and the language model:

  Vision blocks (model.visual.blocks.N), for every N:
    - query proj (vis)   -> slice of attn.qkv
    - key proj (vis)     -> slice of attn.qkv
    - value proj (vis)   -> slice of attn.qkv
    - att output proj    -> attn.proj
    - MLP gate (vis)      -> mlp.gate_proj
    - MLP up (vis)        -> mlp.up_proj
    - MLP down (vis)      -> mlp.down_proj

  Vision-to-language bridge:
    - Vis-to-lan proj     -> visual.merger.mlp.0 and visual.merger.mlp.2

  Language layers (model.language_model.layers.N), for every N:
    - query proj          -> self_attn.q_proj
    - key proj            -> self_attn.k_proj
    - value proj          -> self_attn.v_proj
    - att output proj     -> self_attn.o_proj
    - MLP gate proj       -> mlp.gate_proj
    - MLP up proj         -> mlp.up_proj
    - MLP down proj       -> mlp.down_proj

Each targeted weight matrix gets its own bottom-singular-subspace
(computed once, up front, in float32, under no_grad -- this is a
one-time setup cost, not part of the per-step optimization loop, which
is what keeps the attack loop itself fast). The fused vision qkv
projection is handled by slicing its weight into three row-blocks
(query/key/value) and computing a separate bottom subspace per slice,
while sharing a single forward-pre-hook (the input to qkv is identical
for all three slices, so we avoid registering redundant hooks).

By default every vision block (0-31) and every language layer (0-27)
is included. --chosenLanLayers / --chosenVisLayers can still be
supplied to restrict to a subset, and --includeMerger/--no-includeMerger
toggles the vision-to-language bridge projection.

Example runs:

export CUDA_VISIBLE_DEVICES=3
conda deactivate
cd spectralShift/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
for ATTACK_SAMPLE in $(seq 1 50); do
    python qwen/QwenUntargted_spectSSGRA_AllLayersBlocks.py --attck_type saa_loop --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.5
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python qwen/QwenUntargted_spectSSGRA_AllLayersBlocks.py --attck_type saa_loop --desired_norm_l_inf 0.0035 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.5
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python qwen/QwenUntargted_spectSSGRA_AllLayersBlocks.py --attck_type saa_loop --desired_norm_l_inf 0.0045 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.5
done

# Restrict to a subset of layers if desired, same flags as before:
python qwen/QwenUntargted_spectSSGRA_AllLayersBlocks.py --attck_type saa_loop --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 --attackSample 1 --towardsNull 0.5 --chosenLanLayers 2 --chosenVisLayers 0 1 2 4 5 6 7 8 9 14 24

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


def get_grill_l2(outputs, outputsN):
    loss = 0.0
    for h, hn in zip(outputs.hidden_states, outputsN.hidden_states):
        loss = loss + criterion(h, hn)
    return loss * criterion(h, hn)


def get_grill_wass(outputs, outputsN, startPos, endPos):
    loss = 0.0
    for h, hn in zip(outputs.hidden_states[startPos:endPos], outputsN.hidden_states[startPos:endPos]):
        loss = loss + wasserstein_distance(h, hn)
    return loss


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
    """
    Match Qwen2.5-VL preprocessing closely enough for gradients:
      1. resize with aspect ratio using min_pixels/max_pixels constraints
      2. round dimensions to multiples of patch_size * merge_size

    This returns the resized H/W used before patchification.
    """
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

    # Follow the Qwen-VL smart-resize idea.
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

    Input:
      x01: (1,3,H,W), original image in [0,1]

    Output:
      pixel_values: (num_patches, 3 * temporal_patch_size * patch_size * patch_size)
      image_grid_thw: (1,3), long tensor

    This mirrors Qwen2.5-VL image patchification so gradients flow from model pixel_values
    back to original image space.
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

    # Qwen image processor uses temporal_patch_size=2 for images by duplicating the single frame.
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

    # Same permute as Qwen processor.
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
# All-layer, all-block selected module attack for Qwen2.5-VL
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


# Every language-side linear projection type we attack, keyed by module-name
# suffix -> a human readable label (matches the layer-type legend the user
# supplied, i.e. "query proj ", "key proj ", "value proj ", "MLP gate proj",
# "MLP up proj", "MLP down proj"; att output proj is added too since it is
# the remaining linear layer type present in every language decoder layer).
LANGUAGE_SUFFIX_LABELS = OrderedDict([
    (".self_attn.q_proj", "query proj (lan)"),
    (".self_attn.k_proj", "key proj (lan)"),
    (".self_attn.v_proj", "value proj (lan)"),
    (".self_attn.o_proj", "att output proj (lan)"),
    (".mlp.gate_proj", "MLP gate proj"),
    (".mlp.up_proj", "MLP up proj"),
    (".mlp.down_proj", "MLP down proj"),
])

# Vision-side linear projection types that are NOT the fused qkv matrix
# (qkv is handled separately below since query/key/value share one Linear).
VISION_SUFFIX_LABELS = OrderedDict([
    (".attn.proj", "att output proj (vis)"),
    (".mlp.gate_proj", "MLP gate (vis)"),
    (".mlp.up_proj", "MLP up (vis)"),
    (".mlp.down_proj", "MLP down (vis)"),
])

VISION_QKV_SUFFIX = ".attn.qkv"
VISION_QKV_SUBLABELS = ["query proj (vis)", "key proj (vis)", "value proj (vis)"]

MERGER_NAME_PATTERN = re.compile(r"visual\.merger\.mlp\.(0|2)$")
MERGER_LABEL = "Vis-to-lan proj"


def is_merger_target(name: str) -> bool:
    return bool(MERGER_NAME_PATTERN.search(name))


def collect_target_modules(model, chosen_lan_layers=None, chosen_vis_layers=None, include_merger=True):
    """
    Walks every named module in the model and returns a flat list of
    per-target-slice dicts describing every language, vision, and
    vision-to-language-bridge linear projection to attack:

      {
        "name": unique key used to store this slice's bottom-subspace loss,
        "hook_name": the actual module path a forward-pre-hook is registered on,
        "module": the nn.Linear module,
        "kind": "language" | "vision" | "merger",
        "sub_kind": human readable layer-type label,
        "layer_idx": int or None,
        "weight_slice": (start_row, end_row) or None -- used only for the
                         fused vision qkv projection, which is split into
                         query/key/value row-blocks.
      }
    """
    chosen_lan_layers_set = None if chosen_lan_layers is None else set(chosen_lan_layers)
    chosen_vis_layers_set = None if chosen_vis_layers is None else set(chosen_vis_layers)

    targets = []
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
            for suffix, label in LANGUAGE_SUFFIX_LABELS.items():
                if name.endswith(suffix):
                    targets.append({
                        "name": name,
                        "hook_name": name,
                        "module": module,
                        "kind": "language",
                        "sub_kind": label,
                        "layer_idx": lan_idx,
                        "weight_slice": None,
                    })
                    break
            continue

        # ---------------- vision transformer blocks ----------------
        vis_idx = extract_vision_layer_idx(name)
        if vis_idx is not None and "visual" in name:
            if chosen_vis_layers_set is not None and vis_idx not in chosen_vis_layers_set:
                continue

            if name.endswith(VISION_QKV_SUFFIX):
                out_dim = module.weight.shape[0]
                third = out_dim // 3
                for i, label in enumerate(VISION_QKV_SUBLABELS):
                    targets.append({
                        "name": f"{name}::{label}",
                        "hook_name": name,
                        "module": module,
                        "kind": "vision",
                        "sub_kind": label,
                        "layer_idx": vis_idx,
                        "weight_slice": (i * third, (i + 1) * third),
                    })
                continue

            for suffix, label in VISION_SUFFIX_LABELS.items():
                if name.endswith(suffix):
                    targets.append({
                        "name": name,
                        "hook_name": name,
                        "module": module,
                        "kind": "vision",
                        "sub_kind": label,
                        "layer_idx": vis_idx,
                        "weight_slice": None,
                    })
                    break
            continue

        # ---------------- vision-to-language bridge ----------------
        if include_merger and is_merger_target(name):
            targets.append({
                "name": name,
                "hook_name": name,
                "module": module,
                "kind": "merger",
                "sub_kind": MERGER_LABEL,
                "layer_idx": None,
                "weight_slice": None,
            })

    return targets


def compute_bottom_singular_subspace(weight: torch.Tensor, towardsNull: float, weight_slice=None):
    """
    Same bottom-k selection logic as the original attack:
      absorbSize = len(S[S<1])
      if towardsNull == 0:
          bottomInd = 1
      else:
          bottomInd = int(absorbSize * towardsNull)

    weight_slice, if given, is a (start_row, end_row) tuple used to select
    a row-block of `weight` before the SVD (used for the fused vision qkv
    projection so query/key/value each get their own bottom subspace).
    """
    with torch.no_grad():
        W = weight.detach().to(torch.float32)
        if weight_slice is not None:
            start, end = weight_slice
            W = W[start:end, :]

        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        del U

        absorbSize = int((S < 1).sum().item())

        if towardsNull == 0:
            bottomInd = 1
        else:
            bottomInd = int(absorbSize * towardsNull)

        bottomInd = max(1, bottomInd)
        bottomInd = min(bottomInd, Vh.shape[0])

        V_bottom = Vh[-bottomInd:].contiguous()
        return V_bottom, S, absorbSize, bottomInd


def getMeanAlignmentLossWithBottomSubspace(InputToLayer, bottomRightSingularVectors):
    """
    InputToLayer:
      usually shape (B,T,D), (N,D), or occasionally tuple/list containing tensor.
    bottomRightSingularVectors:
      shape (k,D)
    """
    H = InputToLayer
    if isinstance(H, (tuple, list)):
        H = H[0]

    if H.dim() == 3:
        H = H.reshape(-1, H.shape[-1])
    elif H.dim() == 2:
        pass
    else:
        H = H.view(-1, H.shape[-1])

    V = bottomRightSingularVectors.to(device=H.device, dtype=H.dtype)

    H_hat = F.normalize(H, dim=1)
    V_hat = F.normalize(V, dim=1)

    coeffs = H_hat @ V_hat.T
    per_token_energy = (coeffs ** 2).sum(dim=1)
    loss = ((1.0 - per_token_energy) ** 2).mean()
    return loss


def register_all_hooks(targets):
    """
    Registers exactly one forward-pre-hook per unique underlying module
    (deduplicated by hook_name), even though the fused vision qkv module
    appears three times in `targets` (once per query/key/value slice).
    """
    handles = []
    seen_hook_names = set()
    for spec in targets:
        hook_name = spec["hook_name"]
        if hook_name in seen_hook_names:
            continue
        seen_hook_names.add(hook_name)
        h = spec["module"].register_forward_pre_hook(make_pre_hook(hook_name))
        handles.append(h)
    return handles


def build_target_specs_with_subspaces(model, towardsNull: float, chosen_lan_layers=None, chosen_vis_layers=None, include_merger=True):
    targets = collect_target_modules(
        model,
        chosen_lan_layers=chosen_lan_layers,
        chosen_vis_layers=chosen_vis_layers,
        include_merger=include_merger,
    )
    specs = []

    print("\n========== TARGET MODULES (ALL LAYER TYPES, ALL BLOCKS) ==========")
    for t in targets:
        name = t["name"]
        hook_name = t["hook_name"]
        module = t["module"]
        kind = t["kind"]
        sub_kind = t["sub_kind"]
        layer_idx = t["layer_idx"]
        weight_slice = t["weight_slice"]

        V_bottom, S, absorbSize, bottomInd = compute_bottom_singular_subspace(
            module.weight, towardsNull, weight_slice=weight_slice
        )

        spec = {
            "name": name,
            "hook_name": hook_name,
            "module": module,
            "kind": kind,
            "sub_kind": sub_kind,
            "layer_idx": layer_idx,
            "bottom_vectors": V_bottom,
            "absorbSize": absorbSize,
            "bottomInd": bottomInd,
            "weight_shape": tuple(module.weight.shape),
            "weight_slice": weight_slice,
        }
        specs.append(spec)

        layer_str = f"{layer_idx:3d}" if layer_idx is not None else "N/A"
        print(
            f"{kind:8s} | layer={layer_str} | {sub_kind:24s} | {hook_name} | "
            f"weight_shape={tuple(module.weight.shape)} slice={weight_slice} "
            f"| absorbSize={absorbSize} | bottomInd={bottomInd}"
        )

    num_lang = sum(1 for x in specs if x["kind"] == "language")
    num_vis = sum(1 for x in specs if x["kind"] == "vision")
    num_merger = sum(1 for x in specs if x["kind"] == "merger")
    print("====================================================================")
    print(f"Total language targets: {num_lang}")
    print(f"Total vision targets:   {num_vis}")
    print(f"Total merger targets:   {num_merger}")
    print(f"Total targets overall:  {len(specs)}")
    print("====================================================================\n")

    return specs


def aggregated_bottom_subspace_loss(target_specs, device):
    lang_losses = []
    vis_losses = []
    merger_losses = []

    for spec in target_specs:
        hook_name = spec["hook_name"]
        if hook_name not in layer_inputs:
            continue

        H = layer_inputs[hook_name]
        loss_i = getMeanAlignmentLossWithBottomSubspace(H, spec["bottom_vectors"])

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
    towardsNull: float,
    chosenLanLayers=None,
    chosenVisLayers=None,
    includeMerger=True,
):
    """
    All-layer, all-block version of the original attack:
      - optimize delta in original image space [0,1]
      - differentiably preprocess to Qwen patch pixel_values
      - hook EVERY language/vision/merger linear projection (subject to
        --chosenLanLayers / --chosenVisLayers filtering, if given)
      - minimize alignment loss toward each target's own bottom singular subspace
    """
    x_orig01 = x_orig01.detach().to(device)

    delta = 0.001 * torch.randn_like(x_orig01, device=device)
    delta.requires_grad_(True)

    opt = torch.optim.Adam([delta], lr=lr)

    losses_list = []
    best_loss = 1e20
    best_delta = delta.detach().clone()

    model.eval()

    target_specs = build_target_specs_with_subspaces(
        model,
        towardsNull=towardsNull,
        chosen_lan_layers=chosenLanLayers,
        chosen_vis_layers=chosenVisLayers,
        include_merger=includeMerger,
    )

    if len(target_specs) == 0:
        raise RuntimeError(
            "No target modules found. Check --chosenLanLayers and --chosenVisLayers."
        )

    hook_handles = register_all_hooks(target_specs)

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

        language_loss, vision_loss, merger_loss, total_used = aggregated_bottom_subspace_loss(target_specs, device=device)

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
        description="Qwen2.5-VL ORIGINAL-image-space adversarial attack, ALL language+vision+merger layer types, ALL blocks"
    )
    parser.add_argument("--attck_type", type=str, default="grill_l2", help="kept for compatibility")
    parser.add_argument("--desired_norm_l_inf", type=float, default=0.03, help="epsilon L_inf in ORIGINAL pixel space [0..1]")
    parser.add_argument("--learningRate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--num_steps", type=int, default=None, help="Number of Adam steps")
    parser.add_argument("--numSteps", type=int, default=None, help="Number of Adam steps, old Qwen-style name")
    parser.add_argument("--attackSample", type=str, default="nature", help="which sample")
    parser.add_argument("--AttackStartLayer", type=int, default=0, help="kept for compatibility")
    parser.add_argument("--numLayerstAtAtime", type=int, default=2, help="kept for compatibility")
    parser.add_argument("--towardsNull", type=float, default=0.1, help="same bottom-k selection logic as original attack")

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

    # whichMLP / whichMLPVis are intentionally NOT exposed here: this script
    # always attacks every projection type (attention q/k/v/o and every MLP
    # projection) in every block, so a single-projection selector no longer
    # applies.

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
    towardsNull = float(args.towardsNull)
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

    print("\n[INFO] Qwen ALL-layer, ALL-block variant: attacks EVERY q/k/v/o and MLP")
    print("[INFO] projection in EVERY vision block, EVERY language layer, plus the")
    print("[INFO] vision-to-language merger projection (unless --no-includeMerger).")
    print("[INFO] CLI args --AttackStartLayer and --numLayerstAtAtime are kept for naming/compatibility.")
    print(f"[INFO] chosenLanLayers={chosenLanLayers if chosenLanLayers is not None else 'ALL (0-27)'}")
    print(f"[INFO] chosenVisLayers={chosenVisLayers if chosenVisLayers is not None else 'ALL (0-31)'}")
    print(f"[INFO] includeMerger={includeMerger}")
    print(f"[INFO] towardsNull={towardsNull} uses the same bottom-k selection logic as the original attack.\n")

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
        f"num_steps_{num_steps}_towardsNull_{towardsNull}_AllLayerTypes_lan-{lanLayersTag}_vis-{visLayersTag}_merger-{includeMerger}.npy"
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
        towardsNull=towardsNull,
        chosenLanLayers=chosenLanLayers,
        chosenVisLayers=chosenVisLayers,
        includeMerger=includeMerger,
    )

    adv_img_path = (
        f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
        f"num_steps_{num_steps}_towardsNull_{towardsNull}_AllLayerTypes_lan-{lanLayersTag}_vis-{visLayersTag}_merger-{includeMerger}.png"
    )

    adv_noise_path = (
        f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
        f"num_steps_{num_steps}_towardsNull_{towardsNull}_AllLayerTypes_lan-{lanLayersTag}_vis-{visLayersTag}_merger-{includeMerger}.pt"
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
        f"num_steps_{num_steps}_towardsNull_{towardsNull}_AllLayerTypes_lan-{lanLayersTag}_vis-{visLayersTag}_merger-{includeMerger}.txt"
    )
    with open(advOutTxt, "w") as f:
        f.write(adv_text + "\n")

    print(f"\nSaved outputs to: {advOutTxt}")
    print(f"Saved convergence to: {conv_path}")


if __name__ == "__main__":
    main()
