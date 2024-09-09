from diffusers import StableDiffusionXLControlNetPipeline,StableDiffusionXLControlNetImg2ImgPipeline,ControlNetModel, StableDiffusionXLPipeline,StableDiffusionXLImg2ImgPipeline, AutoencoderKL, DiffusionPipeline
import concurrent.futures
import torch
import argparse
from config import diffusers_config
import os
# ------------------------------- Model Handler ------------------------------ #


class ModelHandler:
    def __init__(self, INPUT_TYPE=None, load_only=False):
        self.vae = None
        self.base = None
        self.prompt = None
        self.canny = None
        self.image2img = None
        self.refiner = None
        self.type = INPUT_TYPE
        self.load_only = load_only
        self.low_memory=torch.cuda.get_device_properties(0).total_memory < 13000000000
        
    def load_model_vae(self):
        print(f"loading vae {diffusers_config['vae']}")
#         if(self.vae is None):
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
                )

    def load_model_canny(self):
        if(self.canny is None):
            self.canny = self.load("canny",ControlNetModel, vae=False, loadOnly=True)
        return self.canny
    def load_model_canny_lora(self):
        return self.load("canny_lora",ControlNetModel, vae=False, loadOnly=True)
    def load(self,name, pipeLineClass,model=None,vae=True,loadOnly=False,**kwargs):
        print(f"loading {name}")
        if(vae):
            kwargs["vae"] = self.load_model_vae()
        else:
            print("No VAE for ",name)
        pipe = pipeLineClass.from_pretrained(model or diffusers_config[name],
                                              **kwargs, **diffusers_config['common_args'])
        if(not loadOnly and not self.load_only and torch.cuda.is_available()):
            pipe.enable_xformers_memory_efficient_attention()
            if(self.low_memory):
                pipe.enable_model_cpu_offload()
            else:
                pipe=pipe.to("cuda")
        return pipe
    
    def load_model_text2img(self,**kwargs):
        return self.load("text2img",StableDiffusionXLPipeline,**kwargs)

    
    def load_model_canny_text2img(self,**kwargs):
        return self.load("text2img", StableDiffusionXLControlNetPipeline, 
                         controlnet=self.load_model_canny(),**kwargs) 
    def load_model_text2img_turbo(self,**kwargs):
        return self.load("turbo",StableDiffusionXLPipeline,**kwargs)
    def load_model_img2img_turbo(self,**kwargs):
        return self.load("turbo",StableDiffusionXLImg2ImgPipeline,**kwargs)
    def load_model_canny_text2img_turbo(self,**kwargs):
        return self.load("turbo",StableDiffusionXLControlNetPipeline, 
                         controlnet=self.load_model_canny(),**kwargs)
    
    def load_model_canny_img2img_turbo(self,**kwargs):
        return self.load("turbo",StableDiffusionXLControlNetImg2ImgPipeline, controlnet=self.load_model_canny(),**kwargs)
        
    def load_model_canny_img2img(self,**kwargs):
        return self.load("img2img",StableDiffusionXLControlNetImg2ImgPipeline, 
                         controlnet=self.load_model_canny(),**kwargs)
    def load_model_img2img(self,**kwargs):
        return self.load("img2img",StableDiffusionXLImg2ImgPipeline,**kwargs)

    def load_model_refiner(self,**kwargs):
        return self.load("refiner",StableDiffusionXLImg2ImgPipeline,**kwargs)

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
            return loaderFn(self,**kwargs)
        else:
            print(f"no callable found with name load_model_{name}")

    def load_models(self, INPUT_TYPE = "canny_img2img"):
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = t-r
        print('Total memory:', t)
        print('Total Available:', a)
        if(self.type == INPUT_TYPE):
            return
        self.base = None
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_base = executor.submit(lambda: self.load_named(INPUT_TYPE))
            if(diffusers_config['use_refiner']):
                self.refiner = None
                future_refiner = executor.submit(self.load_model_refiner)

            self.base = future_base.result()
            self.type = INPUT_TYPE
            if(diffusers_config['use_refiner']):
                self.refiner = future_refiner.result()
            else:
                self.refiner = None



# ---------------------------------- Helper ---------------------------------- #
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model_name", type=str,
                    default="canny_img2img", help="Model name")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    model_name = args.model_name
    if(model_name is None):
        model_name = "canny_img2img"
    MODELS = ModelHandler(load_only=True)
    MODELS.load_named(model_name)
