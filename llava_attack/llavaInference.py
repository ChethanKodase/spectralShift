'''

export CUDA_VISIBLE_DEVICES=3
cd spectralShift/
conda activate llava15
python llava_attack/llavaInference.py

'''

import argparse
import torch
from PIL import Image
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
    CLIPImageProcessor,
    LlamaTokenizer,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/home/luser/LLaVA/llava-1.5-7b-hf")
    ap.add_argument("--image", default="/home/luser/illcond/gemma_attack/dataSamples/astronauts68.jpg")
    ap.add_argument("--question", default="Describe the image in detail.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Build components explicitly (avoids processor_config fields like `image_token`)
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, use_fast=False)
    image_processor = CLIPImageProcessor.from_pretrained(args.model_path)

    processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)

    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if device == "cpu":
        model = model.to(device)


    for name, module in model.named_modules():

        print(name, "->", module.__class__.__name__)

    image = Image.open(args.image).convert("RGB")
    prompt = f"USER: <image>\n{args.question}\nASSISTANT:"

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )

    print(processor.decode(out_ids[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()