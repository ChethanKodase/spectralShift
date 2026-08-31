

'''


export CUDA_VISIBLE_DEVICES=3
conda deactivate
cd spectralShift/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
python qwen/Qwen2p5DetectEachAdversaryPlot.py --attck_type bsa --desired_norm_l_inf 0.005 --thickEpsilon 0.05 --attackMode lan --attackSample 38


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



    args = parser.parse_args()

    attck_type = args.attck_type
    epsilon = float(args.desired_norm_l_inf)
    thickEpsilon = float(args.thickEpsilon)
    attackSample = int(args.attackSample)

    attackMode = str(args.attackMode)
 



    testIt = np.load(f"qwen/allProbMaxes/perAttackSampleProbMaxes_{attackMode}_attck_type_{attck_type}_epsilon_{epsilon}_thickEpsilon_{thickEpsilon}_NumattackSamples_{attackSample}_.npy")

    print("testIt", testIt)

if __name__ == "__main__":
    main()


#perAttackSampleProbMaxes_lan_attck_type_bsa_epsilon_0.005_thickEpsilon_0.05_attackSample_2_
#perAttackSampleProbMaxes_lan_attck_type_bsa_epsilon_0.005_thickEpsilon_0.05_NumattackSamples_2_