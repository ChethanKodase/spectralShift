

'''
export CUDA_VISIBLE_DEVICES=0
cd spectralShift
conda activate share4v
python sharegpt/sharegptLayerCheck.py
'''


import os

import torch
from PIL import Image

from share4v.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from share4v.conversation import conv_templates
from share4v.mm_utils import get_model_name_from_path, tokenizer_image_token
from share4v.model.builder import load_pretrained_model
from share4v.utils import disable_torch_init


# Load model, tokenizer and image processor
MODEL_PATH = "../interpretAttacks/ShareGPT4V/ShareGPT4V-7B"  # or "./checkpoints/ShareGPT4V-7B" if downloaded locally

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

disable_torch_init()
model_name = get_model_name_from_path(MODEL_PATH)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=MODEL_PATH,
    model_base=None,
    model_name=model_name,
    device=device.type,
)
model.eval()
model_dtype = next(model.parameters()).dtype

# Load local image
image_path = "/home/luser/interpretAttacks/llava_attack/dataSamplesForQuant/2.JPEG"



'''for name, param in model.named_parameters():
    print(f"{name:60s} {tuple(param.shape)}")'''


# Written next to this script (sharegpt/model_parameters.txt)
params_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_parameters.txt")
with open(params_out, "w") as f:
    for name, param in model.named_parameters():
        print(f"{name:60s} {tuple(param.shape)}", file=f)
print(f"Saved parameter list to: {params_out}")


try:
    image = Image.open(image_path).convert("RGB")
    print(f"Successfully loaded image: {image_path}")
    print(f"Image size: {image.size}")
except Exception as e:
    print(f"Error loading image: {e}")
    exit(1)


# Prepare prompt (ShareGPT4V conversation template, same as run_share4v.eval_model)
question = "What is shown in this image?"
if getattr(model.config, "mm_use_im_start_end", False):
    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
else:
    qs = DEFAULT_IMAGE_TOKEN + '\n' + question

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
text_prompt = conv.get_prompt()


# Process inputs
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    side = max(width, height)
    result = Image.new(pil_img.mode, (side, side), background_color)
    result.paste(pil_img, ((side - width) // 2, (side - height) // 2))
    return result

if getattr(model.config, "image_aspect_ratio", "pad") == "pad":
    image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))

image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
image_tensor = image_tensor.to(device=model.device, dtype=model_dtype)  # fp16, matches vision tower / mm_projector
#print("image_tensor.shape", image_tensor.shape)

input_ids = tokenizer_image_token(
    text_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
).unsqueeze(0).to(model.device)


# Generate
print("\nGenerating response...")
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        images=image_tensor,
        max_new_tokens=128,
        do_sample=True,  # Optional: for more creative responses
        temperature=0.7,  # Optional: control randomness
        use_cache=True,
    )

# ShareGPT4V/LLaVA-style generate may or may not echo the prompt; strip it if present
gen_ids = generated_ids[0]
if gen_ids.shape[0] > input_ids.shape[1] and torch.equal(gen_ids[:input_ids.shape[1]], input_ids[0]):
    gen_ids = gen_ids[input_ids.shape[1]:]
generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
print("\n=== MODEL RESPONSE ===")
print(generated_text)
