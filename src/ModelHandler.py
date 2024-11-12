from diffusers import StableDiffusionXLControlNetPipeline,StableDiffusionXLControlNetImg2ImgPipeline,ControlNetModel, StableDiffusionXLPipeline,StableDiffusionXLImg2ImgPipeline, AutoencoderKL, DiffusionPipeline, StableDiffusionXLInstructPix2PixPipeline,EDMEulerScheduler
from urllib.request import Request, urlopen
import concurrent.futures
import torch
import argparse
from config import diffusers_config
import pathlib
import os
from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
import requests

from cosxl_edit import set_timesteps_patched
# ------------------------------- Model Handler ------------------------------ #
def isLocalFile(fileName):
    return os.path.isfile(fileName) or pathlib.Path(fileName).suffix == '.safetensors'
def getLocalFile(settings):
    import tqdm
    fileName=settings['model']
    if(not isLocalFile(fileName)):
        print(f"Model {fileName} is not a local file")
        return None
    if not os.path.isfile(fileName):
        url=settings['url']
        print(f"Downloading {url} to {fileName}")
        with requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0',
                'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                'Accept-Encoding':'gzip, deflate, br',
            },allow_redirects=True, stream=True) as r:
            r.raise_for_status()
            with open(fileName, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    #if chunk: 
                    f.write(chunk)
        print(f"Downloaded {fileName}")
    return fileName
        
class ModelHandler:
    def __init__(self, INPUT_TYPE=None, load_only=False):
        self.vae = None
        self.base = None
        self.prompt = None
        self.canny = None
        self.image2img = None
        self.future_refiner = None
        self.future_base = None
        self.refiner = None
        self.type = INPUT_TYPE
        self.load_only = load_only
        self.default_settings = diffusers_config["model_settings"]["default"]
        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.low_memory=not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 13000000000
    
    def __enter__(self):
        self.executor.__enter__()
        return self

    def __exit__(self, *args):
        self.executor.__exit__(*args)
    def load_model_vae(self):
        print(f"loading vae {diffusers_config['vae']}")
#        if(self.vae is None):
# #            AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
#             self.vae = AutoencoderKL.from_pretrained(diffusers_config['vae'], 
#                                                      torch_dtype=torch.float16,
# #                torch_dtype=torch.float16,
# #                use_safetensors=True
#                 )
        return AutoencoderKL.from_pretrained(diffusers_config['vae'], 
                                                     torch_dtype=torch.float16,
#                torch_dtype=torch.float16,
               use_safetensors=True
                ).to("cuda" if torch.cuda.is_available() else "cpu")
#        return self.vae

    def load_model_canny(self):
        if(self.canny is None):
            self.canny = self.load("canny",ControlNetModel, vae=False, loadOnly=True)
        return self.canny
    def load_model_canny_lora(self):
        return self.load("canny_lora",ControlNetModel, vae=False, loadOnly=True)
    def load_to(self,pipe,loadOnly=False):
        if(not loadOnly and not self.load_only and torch.cuda.is_available()):
            if(not torch.cuda.is_available()):
                pipe=pipe.to("cpu")
                return
            pipe.enable_xformers_memory_efficient_attention()
            if(self.low_memory):
                pipe.enable_model_cpu_offload()
            else:
                pipe=pipe.to("cuda")
        return pipe

    def load(self,name, pipeLineClass,model=None,vae=True,loadOnly=False,additional=None,additionalName="",**kwargs):
        print(f"loading {name}")
        settings = diffusers_config["model_settings"]
        settings = self.default_settings | settings.get(name,{})
        if(settings.get("vae", vae)):
            future_vae = self.executor.submit(self.load_model_vae)
        else:
            future_vae = None
            print("No VAE for ",name)
        if(additional):
            future_additional = self.executor.submit(additional)
        else:
            future_additional = None
        if(future_vae):
            kwargs["vae"]=future_vae.result()
        if(future_additional):
            kwargs[additionalName]=future_additional.result()
        
        model = model or settings["model"]
        fileName = getLocalFile(settings)
        if(fileName):
            pipe = pipeLineClass.from_single_file(fileName,
                                            **kwargs, **diffusers_config['common_args'])
        else:
            pipe = pipeLineClass.from_pretrained(model,
                                            **kwargs, **diffusers_config['common_args'])
        if(hasattr(pipe,"scheduler")):
            pipe.scheduler = self.make_scheduler(settings["scheduler"],pipe.scheduler)
        pipe = self.load_to(pipe,loadOnly)
        # if(not loadOnly and not self.load_only and torch.cuda.is_available()):
        #     pipe.enable_xformers_memory_efficient_attention()
        #     if(not torch.cuda.is_available()):
        #         pipe=pipe.to("cpu")
        #     elif(self.low_memory and getattr(pipe,"enable_model_cpu_offload")):
        #         pipe.enable_model_cpu_offload()
        #     else:
        #         pipe=pipe.to("cuda")
        return pipe
    
    def load_model_text2img(self,**kwargs):
        return self.load("text2img",StableDiffusionXLPipeline,**kwargs)

    
    def load_model_canny_text2img(self,**kwargs):
        return self.load("text2img", StableDiffusionXLControlNetPipeline, 
                         additional=self.load_model_canny,additionalName="controlnet",**kwargs) 
    def load_model_text2img_turbo(self,**kwargs):
        return self.load("turbo",StableDiffusionXLPipeline,**kwargs)
    def load_model_img2img_turbo(self,**kwargs):
        return self.load("turbo",StableDiffusionXLImg2ImgPipeline,**kwargs)
    def load_model_canny_text2img_turbo(self,**kwargs):
        return self.load("turbo",StableDiffusionXLControlNetPipeline, 
                         additional=self.load_model_canny,additionalName="controlnet",**kwargs)
    def load_model_canny_text2img_dreamturbo(self,**kwargs):
        return self.load("dreamturbo",StableDiffusionXLControlNetPipeline, additional=self.load_model_canny,additionalName="controlnet",**kwargs)
    def load_model_canny_img2img_dreamturbo(self,**kwargs):
        return self.load("dreamturbo",StableDiffusionXLControlNetImg2ImgPipeline, additional=self.load_model_canny,additionalName="controlnet",**kwargs)
    
    def load_model_canny_img2img_turbo(self,**kwargs):
        return self.load("turbo",StableDiffusionXLControlNetImg2ImgPipeline, 
                         additional=self.load_model_canny,additionalName="controlnet",**kwargs)
        
    def load_model_canny_img2img(self,**kwargs):
        return self.load("img2img",StableDiffusionXLControlNetImg2ImgPipeline, 
                         additional=self.load_model_canny,additionalName="controlnet",**kwargs)
    def load_model_img2img(self,**kwargs):
        return self.load("img2img",StableDiffusionXLImg2ImgPipeline,**kwargs)

    def load_model_refiner(self,**kwargs):
        return self.load("refiner",StableDiffusionXLImg2ImgPipeline,**kwargs)
    
    def load_model_cosxledit(self,**kwargs):
        EDMEulerScheduler.set_timesteps = set_timesteps_patched
        pipe=self.load("cosxledit",StableDiffusionXLInstructPix2PixPipeline,**dict(kwargs, num_in_channels=8,is_cosxl_edit=True))
        pipe.scheduler = EDMEulerScheduler(sigma_min=0.002, sigma_max=120.0, sigma_data=1.0, prediction_type="v_prediction")
        return pipe
    
    def load_all(self):
        for func in dir(ModelHandler):
            if func.startswith("load_model_") and callable(getattr(ModelHandler, func)):
                getattr(self, func)()
        # self.load_text2img()
        # self.load_img2img()
        # self.load_controlnet()
        # self.load_refiner()
    def load_named(self,name,**kwargs):
        loaderFn=getattr(ModelHandler, f"load_model_{name}")
        if(callable(loaderFn)):
            print(f"Found loader {loaderFn}")
            self.type = name
            return loaderFn(self,**kwargs)
        else:
            print(f"no callable found with name load_model_{name}")
    def get_refiner(self):
        if(self.refiner is None):
            if(self.future_refiner is not None):
                self.refiner = self.future_refiner.result()
                self.future_refiner = None
            else:
                self.refiner = self.load("refiner",StableDiffusionXLImg2ImgPipeline)
            self.refiner.text_encoder_2=self.get_base().text_encoder_2
        return self.refiner
    def get_base(self,INPUT_TYPE=None):
        if(not INPUT_TYPE):
            INPUT_TYPE=self.type
        if(self.base is None or self.type != INPUT_TYPE):
            if(self.future_base is not None):
                self.base = self.future_base.result()
                self.future_base = None
                if(self.type != INPUT_TYPE):
                    self.base = None
            self.base = self.load_named(INPUT_TYPE)
        return self.base

    def load_models(self, INPUT_TYPE = "canny_img2img", use_refiner = False):
        if(torch.cuda.is_available()):
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = t-r
            print('Total memory:', t)
            print('Total Available:', a)
        if(self.base and self.type == INPUT_TYPE):
            return
        if(use_refiner and self.refiner is None and self.future_refiner is None):
            self.future_refiner = self.executor.submit(lambda: self.load("refiner",StableDiffusionXLImg2ImgPipeline))
        self.future_base = self.executor.submit(lambda:self.load_named(INPUT_TYPE))
    def make_scheduler(self,name, scheduler):
        if(name is None):
            return scheduler
        schedulerCls=schedulers[name]
        if(scheduler.__class__.__name__ == schedulerCls.__name__):
            return
        s = schedulerCls.from_config(scheduler.config)
        if(hasattr(s.config,"use_karras_sigmas")):
            s.config.use_karras_sigmas=True
            s.config.algorithm_type = 'sde-dpmsolver++'
            print("Using Karras Sigmas")
        return s



schedulers={
        "PNDM": PNDMScheduler,
        "KLMS": LMSDiscreteScheduler,
        "DDIM": DDIMScheduler,
        "K_EULER": EulerDiscreteScheduler,
        "DPMSolverMultistep": DPMSolverMultistepScheduler,
    }

# ---------------------------------- Helper ---------------------------------- #
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model_name", type=str,
                    default="canny_img2img", help="Model name")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    model_name = args.model_name
    if(model_name is None):
        model_name = "canny_text2img"
    MODELS = ModelHandler(load_only=True)
    MODELS.load_named(model_name)
