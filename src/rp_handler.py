import argparse
'''
Contains the handler function that will be called by the serverless.
'''

import os
import base64
import torch
from config   import diffusers_config
from controlnet import predict
from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from controlnet.utils import get_image_from_url
import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from rp_schemas import INPUT_SCHEMA
from ModelHandler import ModelHandler
torch.cuda.empty_cache()

def _save_and_upload_images(images, job_id):
    os.makedirs(f"tmp/{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"tmp/{job_id}/", f"{index}.png")
        image.save(image_path)

        if os.environ.get('BUCKET_ENDPOINT_URL', False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(
                    image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{image_data}")

    if(not diffusers_config["save_to_disk"]):
        rp_cleanup.clean([f"tmp/{job_id}/"])
    return image_urls


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]


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
    job_type=job_input['model_type'] or MODELS.type
    if(job_type and job_type != MODELS.type):
        MODELS.load_models(job_type)
    use_refiner =  job_input['use_refiner'] and MODELS.refiner is not None
    try:
        additional_args = {}
        if starting_image:
            if(job_input['controlnet_type']):
                additional_args['conditioning_scale']=job_input['controlnet_conditioning_scale'] or 0.5
                if(job_type == "canny_text2img"):
                    job_input['image_url'] = None
                print ("Running controlnet")
                controlnet_args = {
                    "model_type": job_input['controlnet_type'],
                    "image_url": starting_image,
                    "image_resolution": job_input['controlnet_image_resolution'],
                    "low_threshold": job_input['controlnet_low_threshold'],
                    "high_threshold": job_input['controlnet_high_threshold'],
                }
                print("Controlnet args",controlnet_args)
                canny_output = predict({"input":controlnet_args,"id":job['id']})
#                if(job_type.find("img2img")>=0):
#                    additional_args['control_image'] = canny_output 
#                    additional_args['image'] = get_image_from_url(starting_image)
#                else:
#                    additional_args['image'] = canny_output
        # Generate latent image using pipe
        if(use_refiner):
            additional_args['denoising_end']=job_input['high_noise_frac'],
            
        print ("Running base model with", additional_args)
        output = MODELS.base(
            prompt=job_input['prompt'],
            negative_prompt=job_input['negative_prompt'],
            height=job_input['height'],
            width=job_input['width'],
            control_image=canny_output if job_type.find("img2img")>=0 else None,
            image= canny_output if job_type.find("img2img")<0 else get_image_from_url(starting_image),
            num_inference_steps=job_input['num_inference_steps'],
            guidance_scale=job_input['guidance_scale'],
            num_images_per_prompt=job_input['num_images'],
            generator=generator,
            output_type= 'pil' if not use_refiner  else 'latent',
            **additional_args
        ).images
        print("Got base output length",len(output))
        if(len(output)>0):
            if(use_refiner):
                print ("Running refiner")
                output = MODELS.refiner(
                    prompt=job_input['prompt'],
                    num_inference_steps=job_input['refiner_inference_steps'],
                    strength=job_input['strength'],
                    image=output,
                    num_images_per_prompt=job_input['num_images'],
                    generator=generator
                ).images
                print("Got retfiner output length",len(output), type(output))
        else:
            return {
                "error": "No output from base model",
                "refresh_worker": False
            }

    except RuntimeError as err:
        return {
            "error": f"RuntimeError: {err}, Stack Trace: {err.__traceback__}",
            "refresh_worker": True
        }
    if(len(output)>0):
        print("Saving output ", type(output[0]))
        image_urls = _save_and_upload_images(output, job['id'])

    results = {
        "images": image_urls,
        # "image_url": image_urls[0],
        "seed": job_input['seed']
    }

    # if starting_image:
    #     results['refresh_worker'] = True

    return results

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model_name", type=str,
                    default="canny_img2img", help="Model URL")
parser.add_argument("--model_type", type=str,
                    default=None, help="Model URL")

args = parser.parse_args()
print(args)
INPUT_TYPE = args.model_name
if(INPUT_TYPE is None):
    INPUT_TYPE = "canny_img2img"
MODELS = ModelHandler()
MODELS.load_models(INPUT_TYPE)

runpod.serverless.start({"handler": generate_image})


