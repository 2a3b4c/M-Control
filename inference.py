from sklearn.decomposition import PCA
from diffusers import ControlNetModel, EulerAncestralDiscreteScheduler, DDIMScheduler, StableDiffusionPipeline, DPMSolverMultistepScheduler, UniPCMultistepScheduler
from diffusers import StableDiffusionControlNetPipelineMControl, StableDiffusionControlNetPipeline
from diffusers.utils import load_image
import torch
import numpy as np
from PIL import Image
import cv2
import os
import random
from torchvision import transforms as T
from math import sqrt
import warnings
warnings.filterwarnings("ignore")
import json

path = "SG161222/Realistic_Vision_V2.0"
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float).to("cuda:0")
pipe_canny = StableDiffusionControlNetPipelineMControl.from_pretrained(path, controlnet=controlnet, local_files_only=True, torch_dtype=torch.float).to("cuda:0")
euler_scheduler = DPMSolverMultistepScheduler.from_config(path, subfolder="scheduler")
# euler_scheduler = DDIMScheduler.from_config(path, subfolder="scheduler")
pipe_canny.scheduler = euler_scheduler
pipe_canny.safety_checker=None

root_path = "./examples/table_depth/"
dir = os.listdir(root_path) 

height = 512
width = 512

for path in dir:
    depth_image = Image.open(root_path + path)
    mask_image = Image.open("./mask_examples/mask_table22.png") 

    generator = torch.Generator(device="cpu").manual_seed(123)
    output_our, _ = pipe_canny(prompt='an apple on the table', c_encoder=3, u_decoder=4, 
                    image=depth_image, inject_step=20, user_local_edit=False, noise_input1=None, 
                    num_inference_steps=30, guidance_scale=9, num_images_per_prompt=1, controlnet_conditioning_scale=float(1),
                    generator=generator, mask=mask_image, height=height, width=width, alpha=0.5, inject_padding_num=5, token_index=[0, 2])
    output_our.images[0].save("./outputs/our.png")
