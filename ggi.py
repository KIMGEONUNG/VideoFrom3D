from omegaconf import OmegaConf
import logging
import argparse
from typing import Literal, Optional
import rp
from einops import rearrange
from peft import LoraConfig
from safetensors.torch import load_file

import torch
from diffusers.utils import load_image
import os
from pathlib import Path

import sys
from glob import glob
from os.path import join
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

sys.path.append('.')

from myutils import ch_channel_cnn
from mypipes.pipe_intp2 import CogVideoXImageToVideoPipeline

logging.basicConfig(level=logging.INFO)


def mix_new_noise(noise, alpha):
    """As alpha --> 1, noise is destroyed"""
    if isinstance(noise, torch.Tensor): return blend_noise(noise, torch.randn_like(noise), alpha)
    elif isinstance(noise, np.ndarray):
        return blend_noise(noise, np.random.randn(*noise.shape), alpha)
    else:
        raise TypeError(
            f"Unsupported input type: {type(noise)}. Expected PyTorch Tensor or NumPy array.")


def blend_noise(noise_background, noise_foreground, alpha):
    """ Variance-preserving blend """
    return (noise_foreground * alpha + noise_background * (1 - alpha)) / (alpha**2 +
                                                                          (1 - alpha)**2)**.5


@torch.no_grad()
def generate_video(
    prompt: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    num_frames: int = 81,
    width: Optional[int] = None,
    height: Optional[int] = None,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    fps: int = 16,
    path_ckpt='',
    noise_warp=None,
    cond_images=None,
    mask_last=None,
    offload=False,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - num_frames (int): Number of frames to generate. CogVideoX1.0 generates 49 frames for 6 seconds at 8 fps, while CogVideoX1.5 produces either 81 or 161 frames, corresponding to 5 seconds or 10 seconds at 16 fps.
    - width (int): The width of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - height (int): The height of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    - fps (int): The frames per second for the generated video.
    """

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    image = None
    video = None

    model_name = model_path.split("/")[-1].lower()

    # HACKED -------------------------------------------------------------------------------------#
    image_start = load_image(image_or_video_path[0])
    image_end = load_image(image_or_video_path[1])
    image = [image_start, image_end]

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
    in_channels = 48  # noise 16 + image 16 + LSD 16
    ch_channel_cnn(pipe.transformer, in_channels)

    transformer_lora_config = LoraConfig(
        r=2048,
        lora_alpha=2048,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    pipe.transformer.add_adapter(transformer_lora_config)
    config = OmegaConf.load(os.path.join(path_ckpt, 'config.json'))
    pipe.transformer.register_to_config(**config)

    shard_files = glob(os.path.join(path_ckpt, '*.safetensors'))
    state_dict = {}
    for shard_file in shard_files:
        tensors = load_file(shard_file)
        state_dict.update(tensors)
    _ = pipe.transformer.load_state_dict(state_dict)
    #---------------------------------------------------------------------------------------------#

    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(lora_path,
                               weight_name="pytorch_lora_weights.safetensors",
                               adapter_name="test_1")
        pipe.fuse_lora(lora_scale=1 / lora_rank)

    # 2. Set Scheduler.
    # pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config,
    #                                                    timestep_spacing="trailing")

    if offload:
        print('enable_sequential_cpu_offload active')
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to('cuda')
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # Handle Cond Images ----------------------------------------------------------------------- #
    VAE_SCALING_FACTOR = pipe.vae.config.scaling_factor

    conds = cond_images.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
    conds = conds.to(dtype=dtype).cuda()
    latent_dist = pipe.vae.encode(conds).latent_dist

    cond_latents = latent_dist.sample() * VAE_SCALING_FACTOR
    cond_latents = cond_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
    # ------------------------------------------------------------------------------------------ #

    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    video_generate = pipe(
        height=height,
        cat_conditions=cond_latents,
        mask_last=mask_last,
        width=width,
        prompt=prompt,
        image=image,
        latents=noise_warp,
        num_inference_steps=num_inference_steps,  # Number of inference steps
        num_frames=num_frames,  # Number of frames to generate
        # use_dynamic_cfg=
        # True,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
    ).frames[0]
    return video_generate


def get_blip_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b",
                                                          torch_dtype=torch.float16)
    model.to(device)

    return model, processor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--model_path", type=str, default="THUDM/CogVideoX-5B-I2V")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--seed", type=int, default=43)

    # hacked ----------------------------------------------------------------------------------- #
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--ignore_mask", default=True)
    parser.add_argument("--offload", action='store_true')
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=1350)
    parser.add_argument("--degradation", type=float, default=0.50)
    parser.add_argument("--target", type=str, required=True)
    # ------------------------------------------------------------------------------------------ #

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    # hacked ----------------------------------------------------------------------------------- #

    ## End point image
    images = [
        Image.open(path).resize((args.width, args.height))
        for path in sorted(glob(join(args.target, 'image_*.png')))
    ]

    if args.prompt is None:
        model, processor = get_blip_model()
        inputs = processor(images=images[0], return_tensors="pt").to('cuda', torch.float16)
        generated_ids = model.generate(**inputs)
        args.prompt = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # for each
    cnt = 0
    start = images[0]
    for i in range(len(images) - 1):
        end = images[i + 1]
        args.image_or_video_path = [start, end]
        name_arc = f'arc{i}'
        path_arc = join(Path(args.target).parent.parent, name_arc)

        ## noise
        noise_path = join(path_arc, 'noises.npy')
        instance_noise = np.load(noise_path)
        instance_noise = torch.tensor(instance_noise)
        instance_noise = rearrange(instance_noise, 'F H W C -> F C H W')
        downtemp_noise = instance_noise[None]
        downtemp_noise = mix_new_noise(downtemp_noise, args.degradation).to(dtype=dtype)

        ## Edge condition
        paths = sorted(glob(join(path_arc, 'hed/*.png')))
        imgs = [Image.open(path).resize((args.width, args.height)) for path in paths]

        cond_images = torch.stack([ToTensor()(img) * 2 - 1 for img in imgs])
        cond_images = rp.resize_list(cond_images, args.num_frames - 3)
        cond_images = torch.cat([cond_images] + [cond_images[[-1]]] * 3)
        cond_images = cond_images[None]

        ## Mask
        mask_last = None
        path_mask = join(path_arc, 'corres_end.png')
        if os.path.exists(path_mask) and not args.ignore_mask:
            print('use mask')
            mask = np.array(Image.open(path_mask))[..., 0] == 0
            mask_last = torch.from_numpy(mask[::4, ::4])
            x_end = np.array(end)[::2, ::2]
            mask = np.array(Image.open(path_mask))[..., [0]] == 0
            x_end = x_end * mask
            x_end = Image.fromarray(x_end)
            x_end.save(join(Path(args.target), f'masked_end_{i}.png'))

        # ------------------------------------------------------------------------------------------ #
        path_ckpt = f'checkpoints/checkpoint-{args.epoch}/transformer'

        args.output_path = join(
            args.target,
            f"d{args.degradation}_e{args.epoch}_n{args.num_inference_steps}",
        )
        os.makedirs(args.output_path, exist_ok=True)

        ## mask_out 저장
        args.image_or_video_path = [start, end]
        videos = generate_video(prompt=args.prompt,
                                model_path=args.model_path,
                                lora_path=args.lora_path,
                                lora_rank=args.lora_rank,
                                output_path=args.output_path,
                                num_frames=args.num_frames,
                                width=args.width,
                                height=args.height,
                                image_or_video_path=args.image_or_video_path,
                                num_inference_steps=args.num_inference_steps,
                                guidance_scale=args.guidance_scale,
                                dtype=dtype,
                                generate_type='t2v',
                                seed=args.seed,
                                fps=args.fps,
                                path_ckpt=path_ckpt,
                                noise_warp=downtemp_noise,
                                cond_images=cond_images,
                                mask_last=mask_last,
                                offload=args.offload)
        for img in videos[:-3]:
            img.save(join(args.output_path, f'{cnt:04}.png'))
            cnt += 1
        start = videos[-1]

