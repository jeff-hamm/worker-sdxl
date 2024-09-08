import torch
import os
base_image = os.environ.get("diffusers__base")
if(not base_image):
    base_image = "stabilityai/stable-diffusion-xl-base-1.0" 
diffusers_config = {
    "vae": "madebyollin/sdxl-vae-fp16-fix",
    "base": base_image,
    "prompt": base_image,
    "canny": "diffusers/controlnet-canny-sdxl-1.0",
    "text2img": base_image,
    "img2img": base_image,
    "turbo": "stabilityai/sdxl-turbo",
    "refiner": "stabilityai/stable-diffusion-xl-refiner-1.0",
    "canny_lora": "https://huggingface.co/stabilityai/control-lora/blob/main/control-LoRAs-rank256/control-lora-canny-rank256.safetensors",
    'common_args': {
        "torch_dtype": torch.float16,
        "use_safetensors": True,
        'variant':"fp16"
    },
    'use_refiner': True
}
for key in diffusers_config:
    envValue = os.environ.get("diffusers__" + key)
    if envValue:
        diffusers_config[key] = envValue