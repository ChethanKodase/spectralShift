


'''
export CUDA_VISIBLE_DEVICES=7
conda deactivate
cd spectralShift/
conda activate gemma3
python gemma_attack/Gemma3_LayerCheck.py

'''




from transformers import Gemma3ForConditionalGeneration, AutoProcessor
import torch
from PIL import Image

# Load model and processor
# Model path / loading convention taken from gemma_attack/gemma3Inference.py.
model_path = "../illcond/gemma_attack/Gemma3-4b"
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
processor = AutoProcessor.from_pretrained(model_path, padding_side="left")

# Load local image (same sample convention as gemma_attack/gemma3Inference.py)
attackSample = 1
image_path = f"../interpretAttacks/gemma_attack/dataSamplesForQuant/{attackSample}.JPEG"


'''for name, param in model.named_parameters():
    print(f"{name:60s} {tuple(param.shape)}")'''


with open("gemma_attack/model_parameters.txt", "w") as f:
    for name, param in model.named_parameters():
        print(f"{name:60s} {tuple(param.shape)}", file=f)


try:
    image = Image.open(image_path).convert("RGB")
    print(f"Successfully loaded image: {image_path}")
    print(f"Image size: {image.size}")
except Exception as e:
    print(f"Error loading image: {e}")
    exit(1)

# Prepare messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"}
        ]
    }
]

#print("image.shape", image.shape)

# Process inputs
text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
inputs = inputs.to(model.device)  # Use model.device instead of hardcoded "cuda"

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
