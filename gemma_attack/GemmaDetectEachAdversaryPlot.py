

'''


export CUDA_VISIBLE_DEVICES=0
conda activate gemma3
cd spectralShift
python gemma_attack/GemmaDetectEachAdversaryPlot.py --attck_type bsa --desired_norm_l_inf 0.005 --thickEpsilon 0.05 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --attackSample 50


export CUDA_VISIBLE_DEVICES=1
conda activate gemma3
cd spectralShift
python gemma_attack/GemmaDetectEachAdversaryPlot.py --attck_type nllm --desired_norm_l_inf 0.005 --thickEpsilon 0.05 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --attackSample 50


export CUDA_VISIBLE_DEVICES=2
conda activate gemma3
cd spectralShift
python gemma_attack/GemmaDetectEachAdversaryPlot.py --attck_type ega --desired_norm_l_inf 0.005 --thickEpsilon 0.05 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --kthSingVec -10 --attackMode lan --ega_ratio 0.2 --attackSample 50


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

    attackMode = str(args.attackMode)

    ega_ratio = float(args.ega_ratio)

    DetProMax = np.load(f"gemma_attack/allProbMaxes/perAttackSampleProbMaxes_{attackMode}_attck_type_{attck_type}_epsilon_{epsilon}_thickEpsilon_{thickEpsilon}_NumattackSamples_{attackSample}_.npy")

    print("testIt", DetProMax)


    print("DetProMax[DetProMax>0.6]", [DetProMax>0.9])

    predictedNum = np.sum([DetProMax>0.6])

    AllNum = len(DetProMax)

    print("DetProMax", DetProMax)
    print("predictedNum", predictedNum)
    print("AllNum", AllNum)


if __name__ == "__main__":
    main()
