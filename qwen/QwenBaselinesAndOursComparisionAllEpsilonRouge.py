'''


export CUDA_VISIBLE_DEVICES=2
conda deactivate
cd spectralShift/
conda activate gemma3
python qwen/QwenBaselinesAndOursComparisionAllEpsilonRouge.py \
    --learningRate 0.001 \
    --num_steps 1000 \
    --AttackStartLayer 0 \
    --numLayerstAtAtime 1 \
    --whichMLP gate_proj \
    --whichMLPVis gate_proj \
    --chosenLanLayers 2 \
    --chosenVisLayers 0 1 2 4 5 6 7 8 9 14 24 \
    --numSamplesConsidered 50\
    --chosenEpsilon 0.005\
    --chosenAttackType bsa

'''

from rouge_score import rouge_scorer
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import FuncFormatter


# ============================================================
# IJCAI-style plotting settings
# ============================================================

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],

    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,

    "lines.linewidth": 2.8,
    "axes.linewidth": 1.0,
})


parser = argparse.ArgumentParser(
    description="Qwen ROUGE-L comparison across perturbation budgets"
)

parser.add_argument("--learningRate", type=float, default=1e-3)
parser.add_argument("--chosenEpsilon", type=float, default=0.005)
parser.add_argument("--num_steps", type=int, default=2000)
parser.add_argument("--AttackStartLayer", type=int, default=0)
parser.add_argument("--AttackStartLayer_vis", type=int, default=0)
parser.add_argument("--numLayerstAtAtime", type=int, default=2)

parser.add_argument("--whichMLP", type=str, default="fc1")
parser.add_argument("--whichMLPVis", type=str, default="fc1")

parser.add_argument("--chosenAttackType", type=str, default="bsa")

parser.add_argument("--numSamplesConsidered", type=int, default=50)

parser.add_argument(
    "--chosenLanLayers",
    type=int,
    nargs="+",
    default=None,
    help="Example: --chosenLanLayers 0 1 2"
)

parser.add_argument(
    "--chosenVisLayers",
    type=int,
    nargs="+",
    default=None,
    help="Example: --chosenVisLayers 11 12 13"
)

args = parser.parse_args()

lr = float(args.learningRate)
chosenEpsilon = float(args.chosenEpsilon)
num_steps = int(args.num_steps)
AttackStartLayer = int(args.AttackStartLayer)
AttackStartLayer_vis = int(args.AttackStartLayer_vis)
numLayerstAtAtime = int(args.numLayerstAtAtime)
whichMLP = str(args.whichMLP)
whichMLPVis = str(args.whichMLPVis)

chosenAttackType = str(args.chosenAttackType)

numSamplesConsidered = int(args.numSamplesConsidered)

chosenLanLayers = args.chosenLanLayers
chosenVisLayers = args.chosenVisLayers

towardsNull = 0.5
ega_ratio = 0.2

#allEpsilons = [0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005]

#allEpsilons = [0.002, 0.003, 0.004]
allEpsilons = [chosenEpsilon]

#all_attck_types = ["bsa", "dra", "fdam", "ssp", "ega", "nllm", "saa_loop", "grill_cosNx", "grill_adv2", "grill_adv3"]
#AllAttckTypes = ["BSA", "DRA", "FDA", "SSPA", "EGA", "CE", "SSPMA\n\SSGRA"]
#AllAttckTypes = ["BSA", "DRA", "FDA", "SSPA", "EGA", "CE", "SSGRA", "NoVanish", "grill_adv", "grill_adv3"]


#all_attck_types = ["bsa", "ega", "nllm"]
#AllAttckTypes = ["BSA", "EGA", "CE"]

all_attck_types = [chosenAttackType]

if chosenAttackType == "bsa":
    chosenAttackTypeName = "BSA"

AllAttckTypes = [chosenAttackTypeName]


# Initialize ROUGE-L scorer
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

precisionMeanForAttacksSeries = []
precisionStdForAttacksSeries = []

recallMeanForAttacksSeries = []
recallStdForAttacksSeries = []

f1MeanForAttacksSeries = []
f1StdForAttacksSeries = []


for epsilon in allEpsilons:

    precisionMeanForAttacks = []
    precisionStdForAttacks = []

    recallMeanForAttacks = []
    recallStdForAttacks = []

    f1MeanForAttacks = []
    f1StdForAttacks = []

    print(f"\nEvaluating epsilon = {epsilon}")

    for attck_type in all_attck_types:

        sampleAggP = []
        sampleAggR = []
        sampleAggF1 = []

        print("attack type:", attck_type)

        for attackSample in range(1, numSamplesConsidered+1):

            if attck_type == "saa_loop":
                advOutputPath = (
                    f"../interpretAttacks/qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
                    f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
                    f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
                    f"num_steps_{num_steps}_towardsNull_{towardsNull}_"
                    f"{whichMLP}_{whichMLPVis}_{chosenLanLayers}_{chosenVisLayers}.txt"
                )
            elif attck_type == "saa_loopC":
                advOutputPath = (
                    f"../interpretAttacks/qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
                    f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
                    f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
                    f"num_steps_{num_steps}_towardsNull_{towardsNull}_{whichMLP}_{whichMLPVis}_{chosenLanLayers}_{chosenVisLayers}.txt"
                    )

            elif attck_type == "ega":
                advOutputPath = (
                    f"../interpretAttacks/qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
                    f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
                    f"num_steps_{num_steps}_ratio_{ega_ratio}.txt"
                )

            else:
                advOutputPath = (
                    f"../interpretAttacks/qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
                    f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
                    f"num_steps_{num_steps}_.txt"
                )



            cleanOutputPath = (
                f"../interpretAttacks/qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
                f"cleanOutput.txt"
            )



            if not os.path.exists(advOutputPath):
                print("Missing adversarial output:", advOutputPath)
                continue

            if not os.path.exists(cleanOutputPath):
                print("Missing clean output:", cleanOutputPath)
                continue

            with open(advOutputPath, "r") as f:
                advOutput = f.read().strip()

            with open(cleanOutputPath, "r") as f:
                cleanOutput = f.read().strip()

            # ROUGE-L: score(reference, hypothesis)
            result = rouge.score(cleanOutput, advOutput)

            sampleAggP.append(result['rougeL'].precision)
            sampleAggR.append(result['rougeL'].recall)
            sampleAggF1.append(result['rougeL'].fmeasure)

        sampleAggP = np.array(sampleAggP)
        sampleAggR = np.array(sampleAggR)
        sampleAggF1 = np.array(sampleAggF1)

        print("sampleAggP.shape", sampleAggP.shape)
        print("sampleAggR.shape", sampleAggR.shape)
        print("sampleAggF1.shape", sampleAggF1.shape)

        np.save(f"qwen/BertRougLists/perAttackSampleROUGEscoresPrecision_attck_type_{attck_type}_epsilon_{epsilon}_NumattackSamples_{attackSample}_.npy", sampleAggP)
        np.save(f"qwen/BertRougLists/perAttackSampleROUGEscoresRecall_attck_type_{attck_type}_epsilon_{epsilon}_NumattackSamples_{attackSample}_.npy", sampleAggR)
        np.save(f"qwen/BertRougLists/perAttackSampleROUGEscoresF1Score_attck_type_{attck_type}_epsilon_{epsilon}_NumattackSamples_{attackSample}_.npy", sampleAggF1)

