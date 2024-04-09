import argparse
parser = argparse.ArgumentParser(description='ip2p-sdxl')
parser.add_argument('--gpuid', '-g', type=str, default='0', required=False)
args = parser.parse_args()

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

import torch
import numpy as np
from PIL import Image
import cv2

from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers import EulerAncestralDiscreteScheduler, UniPCMultistepScheduler
from diffusers import StableDiffusionXLInstructPix2PixPipeline, UNet2DConditionModel

import random
import time


class IMG2IMG:

    def __init__(self, gpu_id):
        self.gpu_id = 'cuda:' + str(gpu_id)

        self.pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
            "./models/ip2p/ip2p_sdxl",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to(self.gpu_id)

        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

    def __call__(self,
                input_image,
                prompt='turn it into tiger',
                n_prompt='',
                height=1024,
                width=1024,
                num_samples=1,
                num_steps=20,
                scale=7.5,
                image_scale=1.5,
                seed=-1,
    ):
        if seed == -1:
            random.seed(time.time())
            seed = random.randint(0, 2147483647)
        print('Initial seed: ' + str(seed))
        generator = torch.manual_seed(seed)

        input_image = input_image.resize((width, height))

        images = self.pipe(
                    image=input_image,
                    prompt=prompt,
                    negative_prompt=n_prompt,
                    generator=generator,
                    num_inference_steps=num_steps,
                    num_images_per_prompt=num_samples,
                    guidance_scale=scale,
                    image_guidance_scale=image_scale,
                    height=height,
                    width=width,
        ).images

        return images


if __name__ == '__main__':
    
    model = IMG2IMG(args.gpuid)
    input_image = Image.open('./images_input/hana.jpg')
    images = model(input_image, prompt='make it a tiger', num_samples=1, seed=-1, num_steps=20, scale=7.5)

    image_idx = 0
    for image in images:
        image.save('./images_output/ip2p_out_' + str(image_idx) + '.png')
        image_idx += 1
