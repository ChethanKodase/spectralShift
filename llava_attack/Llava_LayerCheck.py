


'''
export CUDA_VISIBLE_DEVICES=3
conda deactivate
cd spectralShift/
conda activate llava15
python llava_attack/Llava_LayerCheck.py

'''




from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
    CLIPImageProcessor,
    LlamaTokenizer,
)
import torch
from PIL import Image

# Load model and processor
# Model path / loading convention taken from llava_attack/llavaInference.py:
# components are built explicitly (tokenizer + image_processor -> LlavaProcessor)
# rather than via AutoProcessor, to avoid processor_config fields like
# `image_token` that this LLaVA checkpoint doesn't ship with.
model_path = "/home/luser/LLaVA/llava-1.5-7b-hf"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
image_processor = CLIPImageProcessor.from_pretrained(model_path)
processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)

model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
    low_cpu_mem_usage=True,
)
model.eval()
if device == "cpu":
    model = model.to(device)

# Load local image (same dataSamplesForQuant convention already used for
# llava_attack elsewhere in this repo, e.g. qwen/Qwen2p5FullDistancesOverlap.py)
attackSample = 1
image_path = f"../interpretAttacks/llava_attack/dataSamplesForQuant/{attackSample}.JPEG"


'''for name, param in model.named_parameters():
    print(f"{name:60s} {tuple(param.shape)}")'''


with open("llava_attack/model_parameters.txt", "w") as f:
    for name, param in model.named_parameters():
        print(f"{name:60s} {tuple(param.shape)}", file=f)


try:
    image = Image.open(image_path).convert("RGB")
    print(f"Successfully loaded image: {image_path}")
    print(f"Image size: {image.size}")
except Exception as e:
    print(f"Error loading image: {e}")
    exit(1)

# Prepare prompt (LLaVA-1.5 chat format, same convention as
# llava_attack/llavaInference.py -- no apply_chat_template here, since this
# LlavaProcessor is built from raw components rather than AutoProcessor).
question = "What is shown in this image?"
prompt = f"USER: <image>\n{question}\nASSISTANT:"

#print("image.shape", image.shape)

# Process inputs
inputs = processor(text=prompt, images=image, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Use model.device instead of hardcoded "cuda"

# Generate
print("\nGenerating response...")
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,  # Optional: for more creative responses
        temperature=0.7,  # Optional: control randomness
    )

generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
print("\n=== MODEL RESPONSE ===")
print(generated_texts[0])
