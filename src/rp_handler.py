import asyncio
import json
import argparse
import signal
import sys
import time
from uuid import uuid4
'''
Contains the handler function that will be called by the serverless.
'''

from io import BytesIO
import os
import base64
import torch
from config   import diffusers_config
from controlnet import predict
from controlnet.utils import get_image_from_url
import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.worker import run_job
import numpy as np
from PIL import Image
from rp_schemas import INPUT_SCHEMA
from ModelHandler import ModelHandler
torch.cuda.empty_cache()

DEFAULT_TYPE="cosxledit"

def _get_image_path(job_id,type=None,index=None):
    image_base = os.path.join("/output",time.strftime("%Y%m%d"),job_id)
    # if(type is not None):
    #     image_base = os.path.join(image_base, type)
    os.makedirs(image_base, exist_ok=True)
    
    image_path=''
    image_path+=time.strftime("%H%M%S")
    image_path=f"-{type}" if type is not None else ''
    if(index is not None):
        image_path=f"{index:02}-{image_path}"
    return image_base + ".png"

def _save_image(image,job_id,index,type=None):    
    image_path = _get_image_path(job_id,type)
    image.save(image_path)
    print(f"Saved {type} image {job_id}, ix:{index} to {image_path}")


def _save_and_upload_images(images, job_id,type=None):
    image_urls = []
    image_pils = []
    for index, image in enumerate(images):
        if isinstance(image, torch.Tensor):
            print("Converting tensor to np")
            image = np.asarray(image.cpu() * 255., dtype=np.uint8)
        if isinstance(image, np.ndarray):
           image = Image.fromarray(image)

        _save_image(image,job_id,index,type=None)
        image_pils.append(image)

        if os.environ.get('BUCKET_ENDPOINT_URL', False):
            image_url = rp_upload.upload_image(job_id, _get_image_path(job_id,type))
            image_urls.append(image_url)
        else:
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            image_url = 'data:image/jpeg;base64,' + base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_urls.append(image_url)

    if(not diffusers_config["save_to_disk"]):
        rp_cleanup.clean([os.path.join("/output",job_id)])
    return (image_urls,image_pils)


from nodes.resize import NearestSDXLResolution, ImageScale
from nodes.canny import CannyDetector
sdxlResolutionCalc = NearestSDXLResolution()
resizer = ImageScale()
canny = CannyDetector()

def resize_image_for_sdxl(image_tensor, width, height):
    image_width = image_tensor.size()[2]
    image_height = image_tensor.size()[1]
    if(width > 0 or height > 0):
        print(f"Specified from {image_width}x{image_height} image to {width}x{height}")
        (image_tensor,) = resizer.upscale(image_tensor,"nearest-exact", width, height, "center")
        image_width = image_tensor.size()[2]
        image_height = image_tensor.size()[1]
        return (image_tensor,image_width,image_height)
    (width,height) = sdxlResolutionCalc.op(image_tensor)
    print(f"Resizing from {image_width}x{image_height} image to {width}x{height}")
    (image_tensor,) = resizer.upscale(image_tensor,"nearest-exact", width, height, "center")
    return (image_tensor,width,height)


@torch.inference_mode()
def generate_image(job):
    '''
    Generate an image from text using your Model
    '''
    job_input = job["input"]
    print("Got job input",job)
    job_id = job_input.get('image_id','')
    if(job.get('id',None) is not None):
        if(job_id != ''):
            job_id += "/"
        job_id += job['id'] 
    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    job_input = validated_input['validated_input']

    starting_image = job_input['image_url']

    if job_input['seed'] is None:
        job_input['seed'] = int.from_bytes(os.urandom(2), "big")

    generator = torch.Generator("cuda").manual_seed(job_input['seed'])

    job_type=job_input['model_type'] or MODELS.type
    use_refiner =  job_input['use_refiner']
    if(use_refiner is None): 
        use_refiner = diffusers_config['use_refiner']
    base_model=MODELS.get_base(job_type)
    if(not base_model):
        log.info(f"Model {job_type} not found, using default")
        base_model=MODELS.get_base(DEFAULT_TYPE)
        if(not base_model):
            return {"error": "Default model {DEFAULT_TYPE} not found"}
    base_model.scheduler = MODELS.make_scheduler(
        job_input['scheduler'], base_model.scheduler)
    use_turbo = job_type.find("turbo")>=0
    use_canny = job_type.find("canny")>= 0
    job_types = job_type.split("_")
    base_args = dict()
    try:
        additional_args = {}
        width = job_input['width']
        height = job_input['height']
        if starting_image:
            (starting_image, starting_image_pil) = get_image_from_url(starting_image)
            _save_image(starting_image_pil,job_id,0,"input")
            image_tensor = torch.from_numpy(starting_image)[None,]
            original_width = image_tensor.size()[2]
            original_height = image_tensor.size()[1]

            (starting_image,width,height) = resize_image_for_sdxl(image_tensor,job_input['width'],job_input['height'])
            if(use_canny):
                print ("Running controlnet")
                resolution = diffusers_config['controlnet_image_resolution'] or job_input['controlnet_image_resolution']
                if(not resolution):    
                    resolution = min(starting_image.size()[1], starting_image.size()[2]) 
                
                starting_image = np.asarray(starting_image[0].cpu() * 255., dtype=np.uint8)
                canny_output = [canny(starting_image, output_type="np", 
                              low_threshold=job_input['controlnet_low_threshold'], 
                              high_threshold=job_input['controlnet_high_threshold'],
                                   detect_resolution=resolution)]

                (canny_urls,canny_output)=_save_and_upload_images(canny_output, job_id, "canny")
                # controlnet_args = {
                #     "model_type": job_input['controlnet_type'],
                #     "image_url": starting_image,
                #     "image_resolution": ,
                #     "low_threshold": job_input['controlnet_low_threshold'],
                #     "high_threshold": job_input['controlnet_high_threshold'],
                # }
#                print("Controlnet args",controlnet_args)
#                canny_output = predict({"input":controlnet_args,"id":job['id']})
#                if(job_type.find("img2img")>=0):
#                    additional_args['control_image'] = canny_output 
#                    additional_args['image'] = get_image_from_url(starting_image)
#                else:
#                    additional_args['image'] = canny_output
        # Generate latent image using pipe
        base_args = base_args | dict( 
            prompt=job_input['prompt'],
            negative_prompt=job_input['negative_prompt'],
            height=height or width,
            width=width or height,
            num_inference_steps=job_input['num_inference_steps'],
            guidance_scale=job_input['guidance_scale'],
            num_images_per_prompt=job_input['num_images'],
            generator=generator,
            output_type= 'pil' if not use_refiner  else 'latent',
        )
        if(use_refiner):
            base_args['denoising_end']=job_input['high_noise_frac'],
        # if(original_width and original_height):
        #     base_args['original_size'] = (original_width,original_height)
        if(job_type.find("img2")>=0):
            base_args['strength']=job_input['strength']
        if(use_turbo):
            base_args['guidance_scale'] = 0.0
#            base_args['timestep_spacing']='trailing'
            if(job_type.find("img2")>=0):
                base_args['strength']=0.5
                base_args['num_inference_steps']= 2
            else:
                base_args['num_inference_steps']= 1

        if(use_canny and canny_output):
            base_args['controlnet_conditioning_scale']=job_input['controlnet_conditioning_scale']
            if(job_type.find("img2img")>=0):
                base_args['control_image']=canny_output
                base_args['image'] = Image.fromarray(starting_image)
            else:
                base_args['image'] = canny_output
        elif(job_type.find("img2")>=0 or job_type.find("cosxledit")>=0):
            base_args['image'] = Image.fromarray(np.asarray(starting_image[0].cpu() * 255., dtype=np.uint8))

        if(job_type.find("img2")>=0):
            base_args['strength']=job_input['strength']
        for j in job_types:
            base_args = base_args | diffusers_config['base_settings'].get(j,{})
            callback=diffusers_config['model_settings'].get(j,{}).get('before_run',None)
            if(callable(callback)):
                base_args = base_args | callback(base_args, MODELS.base)
        print ("Running base model with",MODELS.base.__class__.__name__, base_args)

        output = MODELS.base(
            **base_args
        ).images
        print("Got base output length",len(output))
        if(len(output)>0):
            if(use_refiner):
                print ("Running refiner")
                refiner = MODELS.get_refiner()
                output = refiner(
                    prompt=job_input['prompt'],
                    num_inference_steps=base_args["num_inference_steps"] or job_input['refiner_inference_steps'],
                    denoising_start=base_args['denoising_end'],
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
        (image_urls,image_pils) = _save_and_upload_images(output, job_id, "output")

    results = {
        "images": image_urls,
        "args": base_args,
        # "image_url": image_urls[0],
        "seed": job_input['seed']
    }

    # if starting_image:
    #     results['refresh_worker'] = True
#    if(len(image_urls) == 1):
 #       return image_urls[0]
    return results

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model_name", type=str,
                    default=DEFAULT_TYPE, help="Model URL")
parser.add_argument("--model_type", type=str,
                    default=None, help="Model URL")
parser.add_argument("--test_input", type=str,
                    default=None, help="Model URL")
parser.add_argument("--rp_log_level", type=str, default="DEBUG")
parser.add_argument("--rp_debugger",action='store_true',default=False)
parser.add_argument("--rp_serve_api",action='store_true',default=False)
parser.add_argument("--rp_api_port",default=8000)
parser.add_argument("--rp_api_concurrency",default=1)
parser.add_argument("--rp_api_host", type=str, default="localhost",
                    help="Host to start the FastAPI server on.")


args = parser.parse_args()
print(args)
INPUT_TYPE = args.model_name
if(INPUT_TYPE is None):
    INPUT_TYPE = DEFAULT_TYPE
MODELS = ModelHandler()
MODELS.load_models(INPUT_TYPE)


# ---------------------------------------------------------------------------- #
#                            Start Serverless Worker                           #
# ---------------------------------------------------------------------------- #
log = runpod.RunPodLogger()
from fastapi.middleware.cors import CORSMiddleware
def _set_config_args(config) -> dict:
    """
    Sets the config rp_args, removing any recognized arguments from sys.argv.
    Returns: config
    """
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown

    # Directly assign the parsed arguments to config
    config["rp_args"] = vars(args)

    # Parse the test input from JSON
    if config["rp_args"]["test_input"]:
        if(os.path.exists(config["rp_args"]["test_input"])): 
            with open(config["rp_args"]["test_input"], "r", encoding="UTF-8") as file:
                config["rp_args"]["test_input"] = json.loads(file.read())
        # elif(config["rp_args"]["test_input"] and config["rp_args"]["test_input"]):
        #     config["rp_args"]["test_input"] = json.loads(config["rp_args"]["test_input"])

    # Parse the test output from JSON
    if config["rp_args"].get("test_output", None):
        config["rp_args"]["test_output"] = json.loads(config["rp_args"]["test_output"])

    # Set the log level
    if config["rp_args"]["rp_log_level"]:
        log.set_level(config["rp_args"]["rp_log_level"])

    return config
def _signal_handler(sig, frame):
    """
    Handles the SIGINT signal.
    """
    del sig, frame
    log.info("SIGINT received. Shutting down.")
    sys.exit(0)


def configure(config):
    """
    Starts the serverless worker.

    config (Dict[str, Any]): Configuration parameters for the worker.

    config["handler"] (Callable): The handler function to run.

    config["rp_args"] (Dict[str, Any]): Arguments for the worker, populated by runtime arguments.
    """
    print(f"--- Starting Serverless Worker |  Version {runpod.serverless.runpod_version} ---")

    signal.signal(signal.SIGINT, _signal_handler)

    config["reference_counter_start"] = time.perf_counter()
    config = _set_config_args(config)
    return config
async def run_local(config) -> None:
    '''
    Runs the worker locally.
    '''
    # Get the local test job
    if config['rp_args'].get('test_input', None):
        log.info("test_input set, using test_input as job input.")
        local_job = config['rp_args']['test_input']
    else:
        if not os.path.exists("test_input.json"):
            log.warn("test_input.json not found, exiting.")
            sys.exit(1)

        log.info("Using test_input.json as job input.")
        with open("test_input.json", "r", encoding="UTF-8") as file:
            local_job = json.loads(file.read())

    if local_job or local_job.get("input", None) is None:
        log.error("Job has no input parameter. Unable to run.")

    # Set the job ID
    local_job["id"] = local_job.get("id", "local_test")
    log.debug(f"Retrieved local job: {local_job}")

    job_result = await run_job(config["handler"], local_job)


def start(config):

    if config["rp_args"]["rp_serve_api"]:
        log.info("Starting API server.")
        api_server = runpod.serverless.rp_fastapi.WorkerAPI(config)
        
        api_server.rp_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        api_server.start_uvicorn(
            api_host=config['rp_args']['rp_api_host'],
            api_port=config['rp_args']['rp_api_port'],
            api_concurrency=config['rp_args']['rp_api_concurrency']
        )
if __name__ == "__main__":
    if(os.environ.get('IS_OFFLINE', None)):
        config = configure({"handler": generate_image, "rp_args": args})
        if config["rp_args"]["test_input"]:
            log.info("Running test input.")
            asyncio.run(run_local(config))
        start(config)
    else:
        runpod.serverless.start({"handler": generate_image, "rp_args": args})
