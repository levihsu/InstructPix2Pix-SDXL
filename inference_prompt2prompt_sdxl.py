import argparse
parser = argparse.ArgumentParser(description='prompt2prompt-sdxl')
parser.add_argument('--gpuid', '-g', type=str, default='0', required=False)
args = parser.parse_args()

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

import torch
import numpy as np
from PIL import Image
import cv2

import random
import time

from diffusers import StableDiffusionXLPipeline
from diffusers import EulerAncestralDiscreteScheduler, UniPCMultistepScheduler

from pipelines.pipeline_prompt2prompt_sdxl import Prompt2PromptXLPipeline
from metrics.clip_similarity import ClipSimilarity
import torchvision.transforms.functional as F


class TXT2IMG:

    def __init__(self, gpu_id):
        self.gpu_id = 'cuda:' + str(gpu_id)

        self.pipe = Prompt2PromptXLPipeline.from_pretrained(
            "./models/sdxl/sdxlbase_v1",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to(self.gpu_id)

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        self.clip_similarity = ClipSimilarity().to(self.gpu_id)

    def __call__(self,
                prompt='photo of a cute Munchkin cat',
                n_prompt='',
                height=768,
                width=768,
                num_samples=1,
                num_steps=25,
                scale=7.5,
                seed=-1,
                edit_type='refine',
                n_cross_replace=0.8,
                n_self_replace=0.4,
    ):

        while True:

            random.seed(time.time())
            seed = random.randint(0, 2147483647)
            print('Initial seed: ' + str(seed))
            generator = torch.manual_seed(seed)

            np.random.seed(seed + 1)
            scale = 12.0 - np.random.rand() * 5.0
            np.random.seed(seed + 2)
            n_cross_replace = 0.8 - np.random.rand() * 0.2
            np.random.seed(seed + 3)
            n_self_replace = 0.4 - np.random.rand() * 0.2

            print('scale: ' + str(scale))
            print('n_cross_replace: ' + str(n_cross_replace))
            print('n_self_replace: ' + str(n_self_replace))

            latents = torch.randn((1, 4, height//8, width//8), generator=generator).expand(2, 4, height//8, width//8).to(dtype=torch.float16)

            images = self.pipe(prompt,
                        negative_prompt=n_prompt,
                        height=height,
                        width=width,
                        num_inference_steps=num_steps,
                        guidance_scale=scale,
                        num_images_per_prompt=num_samples,
                        generator=generator,
                        latents=latents,
                        cross_attention_kwargs={'edit_type': edit_type, 'n_cross_replace': n_cross_replace, 'n_self_replace': n_self_replace},
            ).images

            source_img = F.pil_to_tensor(images[0]).to(self.gpu_id)
            target_img = F.pil_to_tensor(images[1]).to(self.gpu_id)

            clip_sim_0, clip_sim_1, clip_sim_dir, clip_sim_image = self.clip_similarity(
                source_img[None], target_img[None],
                [prompt[0]], [prompt[1]]
            )

            clip_threshold = 0.12
            clip_dir_threshold = 0.02
            clip_img_threshold = 0.75

            print(clip_sim_0.item())
            print(clip_sim_1.item())
            print(clip_sim_dir.item())
            print(clip_sim_image.item())

            if (clip_sim_0.item() >= clip_threshold) and \
               (clip_sim_1.item() >= clip_threshold) and \
               (clip_sim_dir.item() >= clip_dir_threshold) and \
               (clip_sim_image.item() >= clip_img_threshold):
                
                print('Images qualified')
                break
            else:
                print('Regenerating...')
                       
        return images


if __name__ == '__main__':
    
    model = TXT2IMG(args.gpuid)

    prompt1 = 'a beautiful woman holding an apple'
    prompt2 = 'a beautiful woman holding an orange'
    if len(prompt1.split()) == len(prompt2.split()):
        edit_type = 'replace'
    else:
        edit_type = 'refine'
    
    print(edit_type)

    n_prompt = 'worst quality, low quality, lowres, watermark'

    print(prompt1)
    print(prompt2)

    latents = torch.randn((1, 4, 96, 96)).expand(2, 4, 96, 96).to(dtype=torch.float16)
    images = model(prompt=[prompt1, prompt2], n_prompt=[n_prompt, n_prompt], 
                   num_samples=1, seed=-1, num_steps=20, scale=9.0,
                   edit_type=edit_type, n_cross_replace=-1, n_self_replace=-1)

    image_idx = 0
    for image in images:
        image.save('./images_output/out_' + str(image_idx) + '.png')
        image_idx += 1
