from diffusers import StableDiffusionXLControlNetPipeline,StableDiffusionXLControlNetImg2ImgPipeline,ControlNetModel, StableDiffusionXLPipeline,StableDiffusionXLImg2ImgPipeline, AutoencoderKL, DiffusionPipeline
import concurrent.futures
import torch
# ------------------------------- Model Handler ------------------------------ #
common_args = {
    "torch_dtype": torch.float16,
    "use_safetensors": True,
    'variant':"fp16"
}


class ModelHandler:
    def __init__(self, INPUT_TYPE=None, loadOnly=False):
        self.base = None
        self.prompt = None
        self.controlnet = None
        self.image2img = None
        self.refiner = None
        self.type = None
        self.loadOnly = loadOnly
        
    def load_vae(self):
        return AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", 
            "torch_dtype": torch.float16,
            "use_safetensors": True)
    def load(self,name, pipeLineClass,model,**kwargs):
        print(f"loading {name}")
        pipe = pipeLineClass.from_pretrained(model, **kwargs);
        if(not self.loadOnly or torch.cuda.is_available()):
            pipe.enable_xformers_memory_efficient_attention()
            pipe=pipe.to("cuda", silence_dtype_warnings=True)
        return pipe
    
    def load_prompt(self):
        print("loading prompt")
        base_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", vae=self.load_vae(), 
            **common_args
        )
        return self.to(base_pipe)

    def load_canny_img2img(self):
        StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained();
    
    def load_controlnet(self):
        print("loading controlnet")
        controlnet = ControlNetModel.from_pretrained(
             "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16,
             variant="fp16", use_safetensors=True
        )
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", 
                                            torch_dtype=torch.float16,
                                            use_safetensors=True)
        base_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16,
            variant="fp16", use_safetensors=True
        )
        if(torch.cuda.is_available()):
            base_pipe.enable_xformers_memory_efficient_attention()
            base_pipe=base_pipe.to("cuda", silence_dtype_warnings=True)
        return base_pipe

    def load_img2img(self):
        print("loading img2img")
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", 
                                            torch_dtype=torch.float16,
                                            use_safetensors=True)
        base_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", vae=vae,
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        if(torch.cuda.is_available()):
            base_pipe.enable_xformers_memory_efficient_attention()
            base_pipe=base_pipe.to("cuda", silence_dtype_warnings=True)
        return base_pipe


    def load_refiner(self):
        print("loading refiner")
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", 
                                            torch_dtype=torch.float16,
                                            use_safetensors=True)
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", vae=vae,
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False
        )
        if(torch.cuda.is_available()):
            refiner_pipe.enable_xformers_memory_efficient_attention()
            refiner_pipe=refiner_pipe.to("cuda", silence_dtype_warnings=True)
        return refiner_pipe
    def load_all(self):
        self.load_prompt()
        self.load_img2img()
        self.load_controlnet()
        self.load_refiner()

    def load_models(self, INPUT_TYPE = "controlnet"):
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print('Total memory:', t)
        print('Total Available:', t-a)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            if(INPUT_TYPE == "prompt"):
                future_base = executor.submit(self.load_prompt)
            elif(INPUT_TYPE == "img2img"):
                future_base = executor.submit(self.load_img2img)
            elif(INPUT_TYPE == "controlnet"):
                future_base = executor.submit(self.load_controlnet)
            else:
                future_base = executor.submit(self.load_controlnet)
            if(t  > 13000000000):
                future_refiner = executor.submit(self.load_refiner)

            self.base = future_base.result()
            self.type = INPUT_TYPE
            if(t  > 13000000000):
                self.refiner = future_refiner.result()



# ---------------------------------- Helper ---------------------------------- #


if __name__ == "__main__":
    handler = ModelHandler()
    handler.load_all()
