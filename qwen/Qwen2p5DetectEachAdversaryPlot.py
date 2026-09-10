

'''


export CUDA_VISIBLE_DEVICES=3
conda deactivate
cd spectralShift/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
python qwen/Qwen2p5DetectEachAdversaryPlot.py --attck_type bsa --desired_norm_l_inf 0.002 --thickEpsilon 0.1 --attackMode lan --attackSample 100 --detectionThreshold 0.97 --ignoreThreshold 0.1

python qwen/Qwen2p5DetectEachAdversaryPlot.py --attck_type ega --desired_norm_l_inf 0.005 --thickEpsilon 0.1 --attackMode lan --attackSample 100 --detectionThreshold 0.97 --ignoreThreshold 0.1


python qwen/Qwen2p5DetectEachAdversaryPlot.py --attck_type nllm --desired_norm_l_inf 0.005 --thickEpsilon 0.1 --attackMode lan --attackSample 100 --detectionThreshold 0.97 --ignoreThreshold 0.1


python qwen/Qwen2p5DetectEachAdversaryPlot.py --attck_type justNoise --desired_norm_l_inf 0.005 --thickEpsilon 0.1 --attackMode lan --attackSample 100 --detectionThreshold 0.97 --ignoreThreshold 0.1



export CUDA_VISIBLE_DEVICES=1
conda deactivate
cd spectralShift/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
python qwen/Qwen2p5DetectEachAdversaryPlot.py --attck_type bsa --desired_norm_l_inf 0.002 --thickEpsilon 0.1 --learningRate 0.001 --num_steps 1000 --attackMode lan --detectionThreshold 0.98 --attackSample 100


export CUDA_VISIBLE_DEVICES=2
conda deactivate
cd spectralShift/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
python qwen/Qwen2p5DetectEachAdversaryPlot.py --attck_type nllm --desired_norm_l_inf 0.002 --thickEpsilon 0.1 --learningRate 0.001 --num_steps 1000 --attackMode lan --detectionThreshold 0.9 --attackSample 100


export CUDA_VISIBLE_DEVICES=3
conda deactivate
cd spectralShift/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
python qwen/Qwen2p5DetectEachAdversaryPlot.py --attck_type ega --desired_norm_l_inf 0.002 --thickEpsilon 0.1 --learningRate 0.001 --num_steps 1000 --attackMode lan --detectionThreshold 0.9 --attackSample 100


export CUDA_VISIBLE_DEVICES=0
conda deactivate
cd spectralShift/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
python qwen/Qwen2p5DetectEachAdversaryPlot.py --attck_type justNoise --desired_norm_l_inf 0.003 --thickEpsilon 0.1 --learningRate 0.001 --num_steps 1000 --attackMode lan --detectionThreshold 0.98 --attackSample 100


'''




import os

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
import matplotlib.pyplot as plt


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
    parser.add_argument("--attackSample", type=int, default="nature",
                    help="which sample")
    parser.add_argument("--LanLayerTrack", type=int, default=0,
                        help="whcih language layer you want to talk")

    parser.add_argument("--attackMode", type=str, default="lan",
                    help="Which layer were attacked vis or lan")
    parser.add_argument("--detectionThreshold", type=float, default=0.98,
                        help="Adam learning rate")
    parser.add_argument("--ignoreThreshold", type=float, default=0.1,
                        help="Adam learning rate")


    args = parser.parse_args()

    attck_type = args.attck_type
    epsilon = float(args.desired_norm_l_inf)
    thickEpsilon = float(args.thickEpsilon)
    attackSample = int(args.attackSample)

    attackMode = str(args.attackMode)
    detectionThreshold = float(args.detectionThreshold)
    ignoreThreshold = float(args.ignoreThreshold)



    #DetProMax = np.load(f"qwen/allProbMaxes/ProbMaxes_{attackMode}_attck_type_{attck_type}_epsilon_{epsilon}_thickEpsilon_{thickEpsilon}_NumattackSamples_{attackSample}_detectionThreshold_{detectionThreshold}_ignoreThreshold_{ignoreThreshold}.npy")

    #DetProMin = np.load(f"qwen/allProbMaxes/ProbMins_{attackMode}_attck_type_{attck_type}_epsilon_{epsilon}_thickEpsilon_{thickEpsilon}_NumattackSamples_{attackSample}_detectionThreshold_{detectionThreshold}_ignoreThreshold_{ignoreThreshold}.npy")

    NumTimesYouHitTheMarkPerSample = np.load(f"qwen/allProbMaxes/ChancesYouHit_{attackMode}_attck_type_{attck_type}_epsilon_{epsilon}_thickEpsilon_{thickEpsilon}_NumattackSamples_{attackSample}_detectionThreshold_{detectionThreshold}.npy")

    NumSamples = 168

    print("NumTimesYouHitTheMarkPerSample", NumTimesYouHitTheMarkPerSample)

    chancesYouHitmark = NumTimesYouHitTheMarkPerSample/NumSamples

    NunTruePositives = np.sum(chancesYouHitmark>0.6)

    print("NunTruePositives", NunTruePositives)

if __name__ == "__main__":
    main()


#perAttackSampleProbMaxes_lan_attck_type_bsa_epsilon_0.005_thickEpsilon_0.05_attackSample_2_
#perAttackSampleProbMaxes_lan_attck_type_bsa_epsilon_0.005_thickEpsilon_0.05_NumattackSamples_2_