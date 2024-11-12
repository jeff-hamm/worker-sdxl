from diffusers import StableDiffusionInstructPix2PixPipeline
from prompt_generator import MagifactoryPromptGenerator
import numpy as np
import torch
import math
import random

generator = MagifactoryPromptGenerator()
emotions = [
    "Shock and Disbelief",
    "Fear and Anxiety",
    "Anger and Resentment",
    "Sadness and Grief",
    "Guilt and Shame",
    "Processing and Healing",
    "Confusion and Disorientation",
    "Numbness and Detachment",
    "Resentment and Bitterness",
    "Relief and Gratitude",
    "Hope and Optimism",
    "Strength and Resilience",
    "Forgiveness and Compassion"
]

def generate_prompt(pipeline_args:dict, pipeline:StableDiffusionInstructPix2PixPipeline):
    generated=generator.generate_prompt()
    pipeline_args["prompt"]=generated["prompt"]
    pipeline_args["prompt_2"]=generated["prompt"]
    return pipeline_args
def set_timesteps_patched(self, num_inference_steps: int, device = None):
    self.num_inference_steps = num_inference_steps
    
    ramp = np.linspace(0, 1, self.num_inference_steps)
    sigmas = torch.linspace(math.log(self.config.sigma_min), math.log(self.config.sigma_max), len(ramp)).exp().flip(0)
    
    sigmas = (sigmas).to(dtype=torch.float32, device=device)
    self.timesteps = self.precondition_noise(sigmas)
    
    self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
    self._step_index = None
    self._begin_index = None
    self.sigmas = self.sigmas.to("cpu")