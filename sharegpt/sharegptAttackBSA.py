'''



export CUDA_VISIBLE_DEVICES=3
cd interpretAttacks/
conda activate share4v
python ShareGPT4V/sharegptAttackBSA.py --attck_type bsa --desired_norm_l_inf 0.09 --learningRate 0.001 --num_steps 1000 --attackSample 0 --AttackStartLayer 0 --numLayerstAtAtime 1




export CUDA_VISIBLE_DEVICES=0
cd interpretAttacks/
conda activate share4v
for ATTACK_SAMPLE in $(seq 1 100); do
    python ShareGPT4V/sharegptAttackBSA.py --attck_type bsa --desired_norm_l_inf 0.006 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done
for ATTACK_SAMPLE in $(seq 1 100); do
    python ShareGPT4V/sharegptAttackBSA.py --attck_type bsa --desired_norm_l_inf 0.007 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done
for ATTACK_SAMPLE in $(seq 1 100); do
    python ShareGPT4V/sharegptAttackBSA.py --attck_type bsa --desired_norm_l_inf 0.008 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done
for ATTACK_SAMPLE in $(seq 1 100); do
    python ShareGPT4V/sharegptAttackBSA.py --attck_type bsa --desired_norm_l_inf 0.009 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done



AllAttckTypes = ["BSA", "DRA", "FDA", "SSPA", "EGA", "CE", "SSPMA"]


'''


import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from share4v.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from share4v.conversation import conv_templates
from share4v.mm_utils import get_model_name_from_path, tokenizer_image_token
from share4v.model.builder import load_pretrained_model
from share4v.utils import disable_torch_init


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

criterion = nn.MSELoss()


# ----------------------------
# BSA loss utilities
# ----------------------------
def cos(a, b):
    # Cast up to fp32 before normalize/dot-product. ShareGPT4V runs in fp16
    # (33 language + 24 vision hidden states); this isn't strictly needed
    # by get_bsa_loss/get_bsa_vision_loss below (they use F.cosine_similarity
    # directly, which is numerically well-behaved on its own), but
    # get_bsa_flat_loss/get_bsa_flat_vision_loss do route through this, and
    # keeping it fp32-safe here matches the fix required in the GRILL attack
    # script for the same model.
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()


def cosVis(a, b):
    a = torch.flatten(a).float()
    b = torch.flatten(b).float()
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()


def get_bsa_loss(outputs, outputsN, startPos=None, endPos=None):
    """
    Token-wise cosine similarity over ShareGPT4V (LLaMA/Vicuna) language
    hidden states. The attack minimizes this value directly (no (1-cos)
    transform), matching the Qwen/LLaVA BSA implementation.
    """
    hidden_states = outputs.hidden_states
    clean_hidden_states = outputsN.hidden_states

    if startPos is not None and endPos is not None:
        hidden_states = hidden_states[startPos:endPos]
        clean_hidden_states = clean_hidden_states[startPos:endPos]

    loss = 0.0
    for h, hn in zip(hidden_states, clean_hidden_states):
        cos_per_token = F.cosine_similarity(h.squeeze(0), hn.squeeze(0), dim=1)
        loss = loss + cos_per_token.sum()
    return loss


def get_bsa_flat_loss(outputs, outputsN, startPos=None, endPos=None):
    """
    Flat language-hidden-state BSA objective.
    Returns negative squared cosine distance, same sign convention as Qwen code.
    The attack minimizes this value.
    """
    hidden_states = outputs.hidden_states
    clean_hidden_states = outputsN.hidden_states

    if startPos is not None and endPos is not None:
        hidden_states = hidden_states[startPos:endPos]
        clean_hidden_states = clean_hidden_states[startPos:endPos]

    loss = 0.0
    for h, hn in zip(hidden_states, clean_hidden_states):
        loss = loss + (1.0 - cos(h, hn)) ** 2
    return -1.0 * loss


def get_bsa_vision_loss(acts, actsN):
    """
    Token-wise cosine similarity over CLIP vision tower hidden states.
    """
    loss = 0.0
    for h, hn in zip(acts, actsN):
        cos_per_token = F.cosine_similarity(h, hn, dim=-1)
        loss = loss + cos_per_token.sum()
    return loss


def get_bsa_flat_vision_loss(acts, actsN):
    """
    Flat vision-hidden-state BSA objective.
    """
    loss = 0.0
    for h, hn in zip(acts, actsN):
        loss = loss + (1.0 - cosVis(h, hn)) ** 2
    return -1.0 * loss


# ----------------------------
# Utilities: image <-> tensor
# ----------------------------
def pil_to_tensor01(pil_img: Image.Image) -> torch.Tensor:
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor01_to_pil(t01: torch.Tensor) -> Image.Image:
    if t01.dim() == 4:
        t01 = t01[0]
    t01 = t01.detach().cpu().clamp(0, 1)
    arr = (
        t01.permute(1, 2, 0).numpy() * 255.0
    ).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ----------------------------
# Differentiable preprocessing, CLIP-like
# ----------------------------
def _get_target_hw(image_processor):
    ip = image_processor
    crop = getattr(ip, "crop_size", None)
    size = getattr(ip, "size", None)

    target_h = target_w = None
    if isinstance(crop, dict):
        target_h = crop.get("height", None)
        target_w = crop.get("width", None)
    elif isinstance(crop, int):
        target_h = target_w = crop

    resize_short = None
    if isinstance(size, dict) and "shortest_edge" in size:
        resize_short = size["shortest_edge"]
    elif isinstance(size, dict) and "height" in size and "width" in size:
        resize_short = min(size["height"], size["width"])
    elif isinstance(size, int):
        resize_short = size

    if target_h is None or target_w is None:
        target_h = target_w = 224
    if resize_short is None:
        resize_short = min(target_h, target_w)

    return int(resize_short), int(target_h), int(target_w)


def resize_shortest_edge_keep_aspect(x: torch.Tensor, shortest_edge: int) -> torch.Tensor:
    _, _, H, W = x.shape
    scale = shortest_edge / min(H, W)
    newH = int(round(H * scale))
    newW = int(round(W * scale))
    return F.interpolate(x, size=(newH, newW), mode="bilinear", align_corners=False)


def center_crop(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    _, _, H, W = x.shape
    top = max((H - target_h) // 2, 0)
    left = max((W - target_w) // 2, 0)
    x_crop = x[:, :, top:top + target_h, left:left + target_w]

    pad_h = target_h - x_crop.shape[2]
    pad_w = target_w - x_crop.shape[3]
    if pad_h > 0 or pad_w > 0:
        x_crop = F.pad(x_crop, (0, max(pad_w, 0), 0, max(pad_h, 0)))
    return x_crop


def normalize_like_processor(x01: torch.Tensor, image_processor) -> torch.Tensor:
    mean = torch.tensor(image_processor.image_mean, dtype=x01.dtype, device=x01.device).view(1, 3, 1, 1)
    std = torch.tensor(image_processor.image_std, dtype=x01.dtype, device=x01.device).view(1, 3, 1, 1)
    return (x01 - mean) / std


def expand2square_diff(x01: torch.Tensor, background_color) -> torch.Tensor:
    """
    Differentiable equivalent of share4v.mm_utils.expand2square: pads the
    shorter spatial dimension so the image becomes square, filling with
    background_color (the CLIP image_mean, matching ShareGPT4V's
    image_aspect_ratio == "pad" preprocessing).
    """
    _, C, H, W = x01.shape
    if H == W:
        return x01

    bg = torch.tensor(background_color, dtype=x01.dtype, device=x01.device).view(1, C, 1, 1)
    side = max(H, W)
    out = bg.expand(1, C, side, side).clone()

    if W > H:
        top = (W - H) // 2
        out[:, :, top:top + H, :] = x01
    else:
        left = (H - W) // 2
        out[:, :, :, left:left + W] = x01
    return out


def sharegpt_preprocess_differentiable(
    x01: torch.Tensor,
    image_processor,
    image_aspect_ratio: str = "pad",
    target_dtype: torch.dtype = None,
) -> torch.Tensor:
    x = x01
    if image_aspect_ratio == "pad":
        x = expand2square_diff(x, image_processor.image_mean)

    shortest_edge, th, tw = _get_target_hw(image_processor)
    x = resize_shortest_edge_keep_aspect(x, shortest_edge)
    x = center_crop(x, th, tw)
    x = normalize_like_processor(x, image_processor)

    # ShareGPT4V's CLIPVisionTower casts its *input* to the vision tower's
    # dtype internally, but casts the *output* image_features back to
    # whatever dtype `images` had going in. Handing it fp32 makes
    # encode_images() feed fp32 activations into the fp16 mm_projector
    # Linear and crash on a dtype mismatch, so the model always needs fp16
    # input directly here, same as share4v.eval.run_share4v's `.half()`.
    if target_dtype is not None:
        x = x.to(dtype=target_dtype)
    return x


# ----------------------------
# ShareGPT4V inputs
# ----------------------------
def build_template_inputs(tokenizer, question: str, model, model_path: str, device):
    """
    Builds the ShareGPT4V conversation prompt and tokenizes it with the
    <image> placeholder replaced by IMAGE_TOKEN_INDEX, exactly like
    share4v.eval.run_share4v.eval_model does.
    """
    model_name = get_model_name_from_path(model_path)

    qs = question
    if getattr(model.config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    if 'llama-2' in model_name.lower():
        conv_mode = "share4v_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "share4v_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "share4v_v0"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)

    return {"input_ids": input_ids, "attention_mask": attention_mask}


# ----------------------------
# Generation helper
# ----------------------------
def run_generation_with_images(
    model,
    tokenizer,
    template_inputs,
    images,
    max_new_tokens=128,
):
    model.eval()
    input_ids = template_inputs["input_ids"].clone()

    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            images=images,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    gen_only = out_ids[:, input_ids.shape[1]:]
    return tokenizer.decode(gen_only[0], skip_special_tokens=True).strip()


# ----------------------------
# Vision hooks for ShareGPT4V CLIP tower
# ----------------------------
def get_sharegpt_vision_layers(model):
    """
    ShareGPT4V wraps a HuggingFace CLIPVisionModel inside its own
    CLIPVisionTower module: model.get_vision_tower().vision_tower is the
    actual transformers CLIPVisionModel.
    """
    vt = model.get_vision_tower()
    if hasattr(vt, "vision_tower") and hasattr(vt.vision_tower, "vision_model"):
        enc = vt.vision_tower.vision_model.encoder
        if hasattr(enc, "layers"):
            return enc.layers

    raise RuntimeError("Could not find ShareGPT4V vision tower encoder layers.")


def run_vision_tower_with_hooks(model, images):
    """
    Runs only the ShareGPT4V vision tower and records activations from each
    CLIP encoder layer. These activations are differentiable w.r.t. images.
    """
    acts = []
    handles = []
    layers = get_sharegpt_vision_layers(model)

    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        if torch.is_tensor(out):
            acts.append(out)

    for layer in layers:
        handles.append(layer.register_forward_hook(hook_fn))

    vision_outputs = model.get_vision_tower()(images)

    for h in handles:
        h.remove()

    return vision_outputs, acts


# ----------------------------
# ORIGINAL-image-space BSA attack
# ----------------------------
def adam_attack_original_space(
    model,
    tokenizer,
    image_processor,
    template_inputs,
    x_orig01,
    attck_type: str,
    num_steps: int,
    lr: float,
    epsilon: float,
    device,
    save_conv_path: str,
    image_aspect_ratio: str = "pad",
    AttackStartLayer: int = 0,
    numLayerstAtAtime: int = 1,
):
    x_orig01 = x_orig01.detach().to(device)
    model_dtype = next(model.parameters()).dtype

    delta = 0.001 * torch.randn_like(x_orig01, device=device)
    delta.requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    losses_list = [0.0]
    best_loss = 1e18
    best_delta = delta.detach().clone()

    model.train()
    model.config.use_cache = False
    model.config.output_hidden_states = True
    model.config.return_dict = True

    # Clean outputs and clean vision activations are fixed targets.
    with torch.no_grad():
        pv_clean = sharegpt_preprocess_differentiable(
            x_orig01, image_processor, image_aspect_ratio, target_dtype=model_dtype
        )

        clean_inputs = {
            k: v.clone() if torch.is_tensor(v) else v
            for k, v in template_inputs.items()
        }
        clean_inputs["images"] = pv_clean
        clean_inputs["labels"] = template_inputs["input_ids"]
        clean_inputs["use_cache"] = False

        outputsN = model(
            **clean_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        _, actsN = run_vision_tower_with_hooks(model, pv_clean)

        hidden_len = len(outputsN.hidden_states)
        vision_len = len(actsN)
        print("Number of language hidden states:", hidden_len)
        print("Number of vision hidden states:", vision_len)

        startPos = AttackStartLayer
        endPos = startPos + numLayerstAtAtime
        print("startPos", startPos)
        print("endPos", endPos)

        if endPos > hidden_len:
            raise ValueError(
                f"endPos ({endPos}) exceeds number of language hidden states ({hidden_len})"
            )

    adv_inputs = {
        k: v.clone() if torch.is_tensor(v) else v
        for k, v in template_inputs.items()
    }
    adv_inputs["labels"] = template_inputs["input_ids"]
    adv_inputs["use_cache"] = False

    for step in range(num_steps):
        x_adv01 = (x_orig01 + delta).clamp(0.0, 1.0)
        x_adv01 = torch.max(
            torch.min(x_adv01, x_orig01 + epsilon),
            x_orig01 - epsilon,
        ).clamp(0.0, 1.0)

        pv_adv = sharegpt_preprocess_differentiable(
            x_adv01, image_processor, image_aspect_ratio, target_dtype=model_dtype
        )

        adv_inputs["images"] = pv_adv

        outputs = model(
            **adv_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        _, acts = run_vision_tower_with_hooks(model, pv_adv)


        loss = get_bsa_loss(outputs, outputsN, startPos, endPos) + get_bsa_vision_loss(acts, actsN)


        # Same sign convention as your Qwen implementation: minimize BSA objective.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta.data.clamp_(-epsilon, epsilon)

        lv = float(loss.item())

        if step == 0 or (step + 1) % 10 == 0:
            print(f"[Adam step {step + 1}/{num_steps}] loss={lv:.6f}")

        if lv < best_loss:
            best_loss = lv
            best_delta = delta.detach().clone()
            losses_list.append(lv)
            np.save(save_conv_path, np.array(losses_list, dtype=np.float32))

        del outputs, acts, loss, pv_adv

    with torch.no_grad():
        x_adv01_final = (x_orig01 + best_delta).clamp(0.0, 1.0)
        x_adv01_final = torch.max(
            torch.min(x_adv01_final, x_orig01 + epsilon),
            x_orig01 - epsilon,
        ).clamp(0.0, 1.0)

    return x_adv01_final, best_delta


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ShareGPT4V-7B original-image-space BSA attack"
    )

    parser.add_argument(
        "--attck_type",
        type=str,
        default="bsa",
        help="bsa | bsa_lan | bsa_vis | bsa_flat | bsa_flat_lan | bsa_flat_vis",
    )
    parser.add_argument(
        "--desired_norm_l_inf",
        type=float,
        default=0.01,
        help="epsilon L_inf in original pixel space [0..1]",
    )
    parser.add_argument("--learningRate", type=float, default=1e-3)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--numSteps", type=int, default=None)
    parser.add_argument("--attackSample", type=str, default="astronauts68")
    parser.add_argument("--AttackStartLayer", type=int, default=0)
    parser.add_argument("--numLayerstAtAtime", type=int, default=1)

    args = parser.parse_args()

    attck_type = args.attck_type
    epsilon = float(args.desired_norm_l_inf)
    lr = float(args.learningRate)
    num_steps = int(args.numSteps) if args.numSteps is not None else int(args.num_steps)
    attackSample = str(args.attackSample)
    AttackStartLayer = int(args.AttackStartLayer)
    numLayerstAtAtime = int(args.numLayerstAtAtime)

    MODEL_PATH = "Lin-Chen/ShareGPT4V-7B"  # or "./checkpoints/ShareGPT4V-7B" if downloaded locally

    # Images are ONLY ever read from this directory.
    IMAGE_DIR = "/home/luser/interpretAttacks/llava_attack/dataSamplesForQuant"
    IMAGE_PATH = f"{IMAGE_DIR}/{attackSample}.JPEG"
    QUESTION = "What is shown in this image?"
    MAX_NEW_TOKENS = 128

    # Everything is written under this directory.
    OUTPUT_ROOT = "/home/luser/interpretAttacks/ShareGPT4V"

    os.makedirs(f"{OUTPUT_ROOT}/outputsStorage", exist_ok=True)
    os.makedirs(f"{OUTPUT_ROOT}/outputsStorage/advOutputs/{attackSample}", exist_ok=True)
    os.makedirs(f"{OUTPUT_ROOT}/outputsStorage/convergence/{attackSample}", exist_ok=True)

    conv_path = (
        f"{OUTPUT_ROOT}/outputsStorage/convergence/{attackSample}/"
        f"sharegpt4v_ORIG_attack_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
        f"num_steps_{num_steps}_.npy"
    )

    adv_img_path = (
        f"{OUTPUT_ROOT}/outputsStorage/advOutputs/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
        f"num_steps_{num_steps}_.png"
    )

    adv_noise_path = (
        f"{OUTPUT_ROOT}/outputsStorage/advOutputs/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
        f"num_steps_{num_steps}_.pt"
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    disable_torch_init()

    print("Loading tokenizer + model + image_processor...")
    model_name = get_model_name_from_path(MODEL_PATH)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=MODEL_PATH,
        model_base=None,
        model_name=model_name,
        device=device.type,
    )

    model.eval()
    model.config.use_cache = False

    image_aspect_ratio = getattr(model.config, "image_aspect_ratio", "pad")

    pil = Image.open(IMAGE_PATH).convert("RGB")
    x_orig01 = pil_to_tensor01(pil).to(device)

    template_inputs = build_template_inputs(tokenizer, QUESTION, model, MODEL_PATH, device)

    model_dtype = next(model.parameters()).dtype
    pv_clean = sharegpt_preprocess_differentiable(
        x_orig01, image_processor, image_aspect_ratio, target_dtype=model_dtype
    )

    print("\n=== CLEAN OUTPUT ===")
    clean_text = run_generation_with_images(
        model,
        tokenizer,
        template_inputs,
        pv_clean,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(clean_text)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\nRunning BSA attack...")
    x_adv01, best_pert = adam_attack_original_space(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        template_inputs=template_inputs,
        x_orig01=x_orig01,
        attck_type=attck_type,
        num_steps=num_steps,
        lr=lr,
        epsilon=epsilon,
        device=device,
        save_conv_path=conv_path,
        image_aspect_ratio=image_aspect_ratio,
        AttackStartLayer=AttackStartLayer,
        numLayerstAtAtime=numLayerstAtAtime,
    )

    tensor01_to_pil(x_adv01).save(adv_img_path)
    torch.save(best_pert.detach().cpu(), adv_noise_path)

    print(f"\nSaved ORIGINAL-resolution adversarial image to: {adv_img_path}")
    print(f"Saved perturbation to: {adv_noise_path}")

    pv_adv = sharegpt_preprocess_differentiable(
        x_adv01, image_processor, image_aspect_ratio, target_dtype=model_dtype
    )

    print("\n=== ADVERSARIAL OUTPUT ===")
    adv_text = run_generation_with_images(
        model,
        tokenizer,
        template_inputs,
        pv_adv,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(adv_text)

    cleanOutTxt = f"{OUTPUT_ROOT}/outputsStorage/advOutputs/{attackSample}/cleanOutput.txt"
    with open(cleanOutTxt, "w") as f:
        f.write(clean_text + "\n\n")

    advOutTxt = (
        f"{OUTPUT_ROOT}/outputsStorage/advOutputs/{attackSample}/"
        f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
        f"num_steps_{num_steps}_.txt"
    )
    with open(advOutTxt, "w") as f:
        f.write(adv_text + "\n")

    print(f"\nSaved outputs to: {advOutTxt}")
    print(f"Saved convergence to: {conv_path}")


if __name__ == "__main__":
    main()