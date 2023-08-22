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
import argparse
from omegaconf import OmegaConf

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/test.yaml",
        help="path to the feature extraction config file"
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="height of generating images",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="width of generating images",
    )
    parser.add_argument(
        "--cnet",
        type=str,
        default="lllyasviel/control_v11f1p_sd15_depth",
        help="path to ControlNet model",
    )
    parser.add_argument(
        "--sd",
        type=str,
        default="SG161222/Realistic_Vision_V2.0",
        help="path to StableDiffusion model",
    )

    model_config = parser.parse_args()
    setup_config = OmegaConf.load("./configs/test.yaml")
    print(setup_config)
    # model load
    controlnet = ControlNetModel.from_pretrained(model_config.cnet, torch_dtype=torch.float).to("cuda:0")
    pipe_canny = StableDiffusionControlNetPipelineMControl.from_pretrained(model_config.sd, controlnet=controlnet, local_files_only=True, torch_dtype=torch.float).to("cuda:0")
    euler_scheduler = DPMSolverMultistepScheduler.from_config(model_config.sd, subfolder="scheduler")
    height = model_config.H
    width = model_config.W
    pipe_canny.scheduler = euler_scheduler
    pipe_canny.safety_checker=None
    

    depth_image = Image.open(setup_config.cindition_path)
    mask_image = Image.open(setup_config.mask_path) 
    prompt = setup_config.prompt
    inject_step = setup_config.guidance_threshold
    num_inference_steps = setup_config.num_ddim_sampling_steps
    guidance_scale = setup_config.scale
    alpha = setup_config.alpha
    token_index = setup_config.token_index
    inject_padding_num = setup_config.inject_padding_num
    
    generator = torch.Generator(device="cpu").manual_seed(123)
    output_our, _ = pipe_canny(prompt=prompt, c_encoder=3, u_decoder=4, 
                    image=depth_image, inject_step=inject_step, user_local_edit=False, noise_input1=None, 
                    num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, 
                    num_images_per_prompt=1, controlnet_conditioning_scale=float(1),
                    generator=generator, mask=mask_image, height=height, width=width, alpha=alpha, inject_padding_num=inject_padding_num, token_index=token_index)
    output_our.images[0].save("./outputs/our.png")


if __name__ == "__main__":
    main()