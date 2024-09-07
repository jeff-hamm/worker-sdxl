import argparse
'''
Contains the handler function that will be called by the serverless.
'''

import os
import base64

import torch
from diffusers.utils import load_image
from controlnet import predict
from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from rp_schemas import INPUT_SCHEMA
from ModelHandler import ModelHandler
torch.cuda.empty_cache()

def _save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        if os.environ.get('BUCKET_ENDPOINT_URL', False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(
                    image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{image_data}")

    rp_cleanup.clean([f"/{job_id}"])
    return image_urls


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]


def get_image(image_url, image_base64):
    '''
    Get the image from the provided URL or base64 string.
    Returns a PIL image.
    '''
    if image_url is not None:
        image = rp_download.file(image_url)
        image = image['file_path']

    if image_base64 is not None:
        image_bytes = base64.b64decode(image_base64)
        image = BytesIO(image_bytes)

    input_image = Image.open(image)
    input_image = np.array(input_image)

    return input_image

@torch.inference_mode()
def generate_image(job):
    '''
    Generate an image from text using your Model
    '''
    job_input = job["input"]

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    job_input = validated_input['validated_input']

    starting_image = job_input['image_url']

    if job_input['seed'] is None:
        job_input['seed'] = int.from_bytes(os.urandom(2), "big")

    generator = torch.Generator("cuda").manual_seed(job_input['seed'])

    MODELS.base.scheduler = make_scheduler(
        job_input['scheduler'], MODELS.base.scheduler.config)
    job_input_controlnet = job_input['controlnet']
    job_type=job_input['type'] or MODELS.type
    if(job_type != MODELS.type):
        MODELS.load_models(job_type)
    
    if job_type == "prompt" and starting_image:  # If image_url is provided, run only the refiner pipeline
        init_image = load_image(starting_image).convert("RGB")
        output = MODELS.refiner(
            prompt=job_input['prompt'],
            num_inference_steps=job_input['refiner_inference_steps'],
            strength=job_input['strength'],
            image=init_image,
            generator=generator
        ).images
    else:
        if starting_image:
            job_input['image_url'] = None
            job_input_controlnet['image_url'] = starting_image
            starting_image = predict(job['controlnet'])
        # Generate latent image using pipe
        output = MODELS.base(
            prompt=job_input['prompt'],
            negative_prompt=job_input['negative_prompt'],
            height=job_input['height'],
            width=job_input['width'],
            controlnet_conditioning_scale=(job_input_controlnet and job_input_controlnet['conditioning_scale']) or 0.5, 
            image=starting_image,
            num_inference_steps=job_input['num_inference_steps'],
            guidance_scale=job_input['guidance_scale'],
            denoising_end=job_input['high_noise_frac'],
            output_type="latent",
            num_images_per_prompt=job_input['num_images'],
            generator=generator
        ).images

        try:
            if(MODELS.refiner is not None):
                output = MODELS.refiner(
                    prompt=job_input['prompt'],
                    num_inference_steps=job_input['refiner_inference_steps'],
                    strength=job_input['strength'],
                    image=output,
                    num_images_per_prompt=job_input['num_images'],
                    generator=generator
                ).images
        except RuntimeError as err:
            return {
                "error": f"RuntimeError: {err}, Stack Trace: {err.__traceback__}",
                "refresh_worker": True
            }

    image_urls = _save_and_upload_images(output, job['id'])

    results = {
        "images": image_urls,
        "image_url": image_urls[0],
        "seed": job_input['seed']
    }

    if starting_image:
        results['refresh_worker'] = True

    return results

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input_type", type=str,
                    default="controlnet", help="Model URL")
parser.add_argument("--model_type", type=str,
                    default=None, help="Model URL")

args = parser.parse_args()
print(args)
INPUT_TYPE = args.input_type
if(INPUT_TYPE is None):
    INPUT_TYPE = "controlnet"
MODELS = ModelHandler()
MODELS.load_models(INPUT_TYPE)

runpod.serverless.start({"handler": generate_image})


