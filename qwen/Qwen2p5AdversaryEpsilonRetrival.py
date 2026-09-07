

'''


---------------------------------------------------------------------------

export CUDA_VISIBLE_DEVICES=0
conda deactivate
cd spectralShift/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
python qwen/Qwen2p5DetectEachAdversary.py --attck_type bsa --desired_norm_l_inf 0.001 --thickEpsilon 0.005 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --detectionThreshold 0.97 --ignoreThreshold 0.1


export CUDA_VISIBLE_DEVICES=1
conda deactivate
cd spectralShift/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
python qwen/Qwen2p5AdversaryEpsilonRetrival.py --attck_type nllm --desired_norm_l_inf 0.001 --thickEpsilon 0.001 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --detectionThreshold 0.97 --ignoreThreshold 0.1


export CUDA_VISIBLE_DEVICES=2
conda deactivate
cd spectralShift/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
python qwen/Qwen2p5AdversaryEpsilonRetrival.py --attck_type ega --desired_norm_l_inf 0.001 --thickEpsilon 0.001 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --detectionThreshold 0.97 --ignoreThreshold 0.1


export CUDA_VISIBLE_DEVICES=2
conda deactivate
cd spectralShift/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
python qwen/Qwen2p5AdversaryEpsilonRetrival.py --attck_type justNoise --desired_norm_l_inf 0.001 --learningRate 0.001 --num_steps 1000 


 : BSA Adv local var 

epsilon        : justNoise local var   : BSA local var              :        NLL local var          :   EGA local var 

epsilon 0.001 : 0.019627340137958527   : 0.019666291773319244       :   0.019666291773319244        :   0.019668713212013245

epsilon 0.002 : 0.019659366458654404   : 0.019718315452337265       :   0.019718315452337265        :   0.01973733678460121

epsilon 0.003 : 0.01969139091670513    : 0.01986018754541874        :   0.01986018754541874         :   0.019885443150997162

epsilon 0.004 : 0.01972348801791668    : 0.02003267966210842        :   0.02003267966210842         :   0.020071856677532196

epsilon 0.005 : 0.01975635066628456    : 0.020276807248592377         :   0.020276807248592377        :   0.02022310346364975



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
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import matplotlib.pyplot as plt


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






# ----------------------------
# Utilities: image <-> tensor
# ----------------------------
def pil_to_tensor01(pil_img: Image.Image) -> torch.Tensor:
    """PIL RGB -> torch float tensor in [0,1], shape (1,3,H,W)"""
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    return t




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


def getImageLocalVar(x_ad):
    dx = x_ad[:, :, :, 1:] - x_ad[:, :, :, :-1]
    dy = x_ad[:, :, 1:, :] - x_ad[:, :, :-1, :]
    ObLocVar = (dx.abs().mean() + dy.abs().mean()) / 2
    return ObLocVar


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


def median_filter_3x3(x):
    # x: [B, C, H, W]

    x_pad = F.pad(
        x,
        (1, 1, 1, 1),
        mode="reflect"
    )

    patches = x_pad.unfold(2, 3, 1).unfold(3, 3, 1)
    # [B, C, H, W, 3, 3]

    patches = patches.contiguous().view(
        *patches.shape[:4],
        9
    )
    # [B, C, H, W, 9]

    sorted_values = torch.sort(
        patches,
        dim=-1
    ).values

    # Median of 9 values = 5th value
    return sorted_values[..., 4]


def smooth_3x3(x):
    x_pad = F.pad(
        x,
        (1, 1, 1, 1),
        mode="reflect"
    )

    return F.avg_pool2d(
        x_pad,
        kernel_size=3,
        stride=1
    )


def polynomial_reconstruct(x, patch_size=5, degree=2):
    """
    Reconstruct each pixel using a local 2D polynomial fitted to
    neighboring pixels, excluding the center pixel.

    Parameters
    ----------
    x : torch.Tensor
        Image tensor of shape [B, C, H, W].

    patch_size : int
        Odd patch size, e.g. 3, 5, 7, 9.

    degree : int
        Degree of the 2D polynomial, e.g. 1, 2, 3, 4.

    Returns
    -------
    reconstructed : torch.Tensor
        Reconstructed image with same shape as x.
    """

    assert patch_size % 2 == 1, "patch_size must be odd"
    assert degree >= 0, "degree must be >= 0"

    B, C, H, W = x.shape
    device = x.device
    dtype = x.dtype

    radius = patch_size // 2

    # ---------------------------------------------------------
    # Generate patch coordinates
    # ---------------------------------------------------------
    yy, xx = torch.meshgrid(
        torch.arange(
            -radius,
            radius + 1,
            device=device,
            dtype=torch.float32
        ),
        torch.arange(
            -radius,
            radius + 1,
            device=device,
            dtype=torch.float32
        ),
        indexing="ij"
    )

    xx = xx.flatten()
    yy = yy.flatten()

    # ---------------------------------------------------------
    # Remove center pixel (0,0)
    # ---------------------------------------------------------
    keep = ~((xx == 0) & (yy == 0))

    xx_neighbors = xx[keep]
    yy_neighbors = yy[keep]

    num_neighbors = xx_neighbors.numel()

    # ---------------------------------------------------------
    # Generate all polynomial powers:
    #
    # degree=1:
    # (0,0), (1,0), (0,1)
    #
    # degree=2:
    # additionally
    # (2,0), (1,1), (0,2)
    #
    # degree=3:
    # additionally
    # (3,0), (2,1), (1,2), (0,3)
    # ---------------------------------------------------------
    powers = []

    for total_degree in range(degree + 1):
        for x_power in range(total_degree + 1):

            y_power = total_degree - x_power

            powers.append(
                (x_power, y_power)
            )

    num_coefficients = len(powers)

    # Number of coefficients in a 2D polynomial:
    #
    # (degree+1)(degree+2)/2
    #
    if num_neighbors < num_coefficients:
        raise ValueError(
            f"Not enough neighboring pixels for degree {degree}. "
            f"patch_size={patch_size} provides {num_neighbors} "
            f"neighbors, but degree={degree} requires "
            f"{num_coefficients} polynomial coefficients."
        )

    # ---------------------------------------------------------
    # Construct design matrix A
    #
    # Each column corresponds to x^i * y^j
    # ---------------------------------------------------------
    columns = []

    for x_power, y_power in powers:

        column = (
            xx_neighbors.pow(x_power)
            *
            yy_neighbors.pow(y_power)
        )

        columns.append(column)

    A = torch.stack(columns, dim=1)

    # A shape:
    #
    # [number_of_neighbors, number_of_coefficients]

    # ---------------------------------------------------------
    # Compute least-squares pseudoinverse
    # ---------------------------------------------------------
    A_pinv = torch.linalg.pinv(A)

    # ---------------------------------------------------------
    # Polynomial basis evaluated at center (x=0,y=0)
    #
    # Since all nonconstant terms become zero:
    #
    # [1, 0, 0, 0, ...]
    # ---------------------------------------------------------
    center_basis = torch.zeros(
        num_coefficients,
        device=device,
        dtype=torch.float32
    )

    center_basis[0] = 1.0

    # Direct interpolation weights
    #
    # Instead of fitting coefficients independently for every
    # pixel, calculate weights once.
    weights = center_basis @ A_pinv

    weights = weights.to(dtype)

    # ---------------------------------------------------------
    # Extract all image patches
    # ---------------------------------------------------------
    xp = F.pad(
        x,
        (radius, radius, radius, radius),
        mode="reflect"
    )

    patches = F.unfold(
        xp,
        kernel_size=patch_size
    )

    # patches:
    # [B, C * patch_size^2, H*W]

    patches = patches.view(
        B,
        C,
        patch_size * patch_size,
        H * W
    )

    # ---------------------------------------------------------
    # Remove center pixel
    # ---------------------------------------------------------
    center_idx = (
        patch_size * patch_size
    ) // 2

    neighbors = torch.cat(
        [
            patches[:, :, :center_idx, :],
            patches[:, :, center_idx + 1:, :]
        ],
        dim=2
    )

    # neighbors:
    # [B, C, patch_size^2 - 1, H*W]

    # ---------------------------------------------------------
    # Reconstruct center pixels
    # ---------------------------------------------------------
    reconstructed = (
        neighbors
        *
        weights.view(1, 1, -1, 1)
    ).sum(dim=2)

    reconstructed = reconstructed.view(
        B,
        C,
        H,
        W
    )

    return reconstructed.clamp(0.0, 1.0)


def pca_reconstruct(x, patch_size=8, num_components=20):
    """
    PCA reconstruction of an image using non-overlapping patches.

    x:
        [B, C, H, W], expected in [0,1]

    patch_size:
        spatial patch size, e.g. 4, 8, 16

    num_components:
        number of PCA components retained
    """

    B, C, H, W = x.shape

    assert B == 1, "This implementation assumes batch size = 1"

    # --------------------------------------------------
    # Crop so H and W are divisible by patch_size
    # --------------------------------------------------

    H_crop = (H // patch_size) * patch_size
    W_crop = (W // patch_size) * patch_size

    x_crop = x[:, :, :H_crop, :W_crop]

    # --------------------------------------------------
    # Extract patches
    # --------------------------------------------------

    patches = F.unfold(
        x_crop,
        kernel_size=patch_size,
        stride=patch_size
    )

    # [1, C*patch_size*patch_size, N]

    patches = patches.squeeze(0).T

    # [N, D]
    #
    # D = C * patch_size * patch_size

    original_dtype = patches.dtype

    # SVD/PCA is more reliable in float32
    patches = patches.float()

    # --------------------------------------------------
    # Center data
    # --------------------------------------------------

    mean_patch = patches.mean(dim=0, keepdim=True)

    X = patches - mean_patch

    # --------------------------------------------------
    # PCA through SVD
    # --------------------------------------------------

    U, S, Vh = torch.linalg.svd(
        X,
        full_matrices=False
    )

    k = min(
        num_components,
        Vh.shape[0]
    )

    V = Vh[:k]

    # --------------------------------------------------
    # Project onto top-k PCA directions
    # --------------------------------------------------

    coefficients = X @ V.T

    # --------------------------------------------------
    # Reconstruct
    # --------------------------------------------------

    X_reconstructed = coefficients @ V

    reconstructed_patches = (
        X_reconstructed + mean_patch
    )

    # --------------------------------------------------
    # Put patches back into image
    # --------------------------------------------------

    reconstructed_patches = (
        reconstructed_patches.T
        .unsqueeze(0)
        .to(original_dtype)
    )

    x_crop_reconstructed = F.fold(
        reconstructed_patches,
        output_size=(H_crop, W_crop),
        kernel_size=patch_size,
        stride=patch_size
    )

    # --------------------------------------------------
    # Handle dimensions that were cropped
    # --------------------------------------------------

    x_reconstructed = x.clone()

    x_reconstructed[
        :, :, :H_crop, :W_crop
    ] = x_crop_reconstructed

    return x_reconstructed.clamp(0, 1)


def tensor01_to_pil(t01: torch.Tensor) -> Image.Image:
    """torch tensor [0,1], shape (1,3,H,W) or (3,H,W) -> PIL RGB"""
    if t01.dim() == 4:
        t01 = t01[0]
    t01 = t01.detach().cpu().clamp(0, 1)
    arr = (t01.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def adam_attack_original_space(
    processor,
    template_inputs,
    x_orig01,               # (1,3,H0,W0) in [0,1]
    attck_type: str,
    epsilon: float,         # L_inf bound in ORIGINAL pixel space [0,1]
    device,
    best_delta
):

    x_orig01 = x_orig01.detach().to(device)


    delta = best_delta
    delta.requires_grad_(True)


    best_delta = delta.detach().clone()


    with torch.no_grad():
        pv_clean_fixed, grid_clean_fixed = qwen_preprocess_differentiable(x_orig01, processor)

        clean_inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
        clean_inputs["pixel_values"] = pv_clean_fixed
        clean_inputs["image_grid_thw"] = grid_clean_fixed
        clean_inputs["labels"] = template_inputs["input_ids"]
        clean_inputs["use_cache"] = False

    adv_inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
    adv_inputs["labels"] = template_inputs["input_ids"]
    adv_inputs["use_cache"] = False

    if attck_type == "justNoise":
        checkItthatWay = torch.randn_like(x_orig01)
        checkItthatWayNormal = 2 * (checkItthatWay - checkItthatWay.min()) / (checkItthatWay.max() - checkItthatWay.min()) - 1

        justNoiseHere = checkItthatWayNormal * epsilon

        x_adv01 = (x_orig01 + justNoiseHere).clamp(0.0, 1.0)
        x_adv01 = torch.max(torch.min(x_adv01, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

        
        observed_local_var = getImageLocalVar(x_adv01)

        obLocVarListArr, AllEpsilon = getImageLocalVarListPerPerturbation(x_orig01, Start=0.001, End=0.005, NumPts=5)
        slope, intercept = np.polyfit(AllEpsilon, obLocVarListArr, 1)
        retrivedEpsilon = (observed_local_var - intercept) / slope
        print("retrivedEpsilon by x_orig01", retrivedEpsilon)
        #print("observed_local_var", observed_local_var)


        #x_reconstructed = median_filter_3x3(x_adv01)
        #x_reconstructed = smooth_3x3(x_adv01)

        '''x_reconstructed = polynomial_reconstruct(
            x_adv01,
            patch_size=5,
            degree=2
        )'''

        x_reconstructed = pca_reconstruct(
            x_adv01,
            patch_size=4,
            num_components=3
        )
        observed_local_var = getImageLocalVar(x_adv01)
        obLocVarListArr, AllEpsilon = getImageLocalVarListPerPerturbation(x_reconstructed, Start=0.001, End=0.005, NumPts=5)
        slope, intercept = np.polyfit(AllEpsilon, obLocVarListArr, 1)
        retrivedEpsilon = (observed_local_var - intercept) / slope
        print("retrivedEpsilon by x_reconstructed", retrivedEpsilon)
        #print("observed_local_var", observed_local_var)


        # Example: polynomial/PCA/smoothed reconstruction
        residual = x_adv01 - x_reconstructed

        estimated_strength_max = residual.abs().max()
        estimated_strength_rms = 2*residual.pow(2).mean()#.sqrt()
        estimated_strength_mae = residual.abs().mean()
        print("estimated_strength_max", estimated_strength_max)
        print("estimated_strength_rms", estimated_strength_rms)
        print("estimated_strength_mae", estimated_strength_mae)

    return retrivedEpsilon, x_adv01, x_reconstructed


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL ORIGINAL-image-space adversarial attack (no squeeze)")
    parser.add_argument("--attck_type", type=str, default="bsa",
                        help="bsa | bsa_flat | bsa_flat_lan | bsa_flat_vis")
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
    parser.add_argument("--detectionThreshold", type=float, default=0.98,
                        help="Adam learning rate")
    parser.add_argument("--ignoreThreshold", type=float, default=0.1,
                        help="Adam learning rate")

    args = parser.parse_args()

    attck_type = args.attck_type
    epsilon = float(args.desired_norm_l_inf)
    lr = float(args.learningRate)
    num_steps = int(args.num_steps)
    attackSample = int(args.attackSample)




    MODEL_PATH = "../illcond/QwenAttack/Qwen2.5-VL-7B-Instruct"
    QUESTION = "What is shown in this image?"
    MAX_NEW_TOKENS = 128

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"device={device}, dtype={dtype}")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)



    for attackSample in range(1,101):


        if attackSample == 39 or attackSample == 58:
            IMAGE_PATH = f"../interpretAttacks/llava_attack/dataSamplesForQuant/{attackSample}s.JPEG"        
        else:
            IMAGE_PATH = f"../interpretAttacks/llava_attack/dataSamplesForQuant/{attackSample}.JPEG"



        pil = Image.open(IMAGE_PATH).convert("RGB")
        x_orig01 = pil_to_tensor01(pil).to(device)

        template_inputs = build_template_inputs(processor, QUESTION, pil, device)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        if attck_type=="bsa":
            adv_noise_path = (
                f"../interpretAttacks/qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
                f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.pt"
            )

        if attck_type=="nllm":
            adv_noise_path = (
                f"../interpretAttacks/qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
                f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.pt"
            )

        if attck_type == "ega":
            ega_ratio = 0.2
            adv_noise_path = (
                f"../interpretAttacks/qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
                f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_ratio_{ega_ratio}.pt"
            )
        else:
            attck_typeTemp = "bsa"
            adv_noise_path = (
                f"../interpretAttacks/qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
                f"adv_ORIG_attackType_{attck_typeTemp}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.pt"
            )

        best_delta = torch.load(adv_noise_path, map_location=device).to(device=device, dtype=x_orig01.dtype) #* 0

        print("attackSample", attackSample)
        print()
        retrivedEpsilon, x_adv01, x_reconstructed = adam_attack_original_space(
            processor=processor,
            template_inputs=template_inputs,
            x_orig01=x_orig01,
            attck_type=attck_type,
            epsilon=epsilon,
            device=device,
            best_delta = best_delta
        )


        save_dir = "qwen/AllPlots/AtestPlot"
        os.makedirs(save_dir, exist_ok=True)

        adv_pil = tensor01_to_pil(x_adv01)
        adv_pil.save(
            os.path.join(save_dir, f"x_adv_{attackSample}.jpeg"),
            format="JPEG",
            quality=95
        )

        adv_pil = tensor01_to_pil(x_reconstructed)
        adv_pil.save(
            os.path.join(save_dir, f"x_recon{attackSample}.jpeg"),
            format="JPEG",
            quality=95
        )

if __name__ == "__main__":
    main()
