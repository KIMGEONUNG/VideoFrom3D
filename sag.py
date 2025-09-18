import sys
import random
from pathlib import Path
import cv2

sys.path.append('.')
import argparse
from PIL import Image
import os
from src.flux.xflux_pipeline import XFluxPipeline
import numpy as np
import torch
from einops import rearrange
from types import MethodType
from src.flux.sampling import denoise, denoise_controlnet, get_noise, get_schedule, prepare, unpack
import os
from os.path import join
from glob import glob
from socket import gethostname
from src.flux.util import load_checkpoint, get_lora_rank
from src.flux.modules.layers import (
    SingleStreamBlockProcessor,
    DoubleStreamBlockProcessor,
    SingleStreamBlockLoraProcessor,
    DoubleStreamBlockLoraProcessor,
)
import torch.nn.functional as F
import hashlib


def checksum_string(data: str, algorithm: str = 'sha256') -> str:
    hash_func = hashlib.new('sha256')
    hash_func.update(data.encode('utf-8'))
    return hash_func.hexdigest()


USE_ELSE_NOISE = False
COUNTER = 0
NUM_REPLACEMENT = None


def forward(
    self,
    warped_x,
    mask,
    prompt,
    width,
    height,
    guidance,
    num_steps,
    seed,
    controlnet_image=None,
    timestep_to_start_cfg=0,
    true_gs=3.5,
    control_weight=0.9,
    neg_prompt="",
    image_proj=None,
    neg_image_proj=None,
    ip_scale=1.0,
    neg_ip_scale=1.0,
):
    global USE_ELSE_NOISE
    global COUNTER
    global NUM_REPLACEMENT

    x = get_noise(1, height, width, device=self.device, dtype=torch.bfloat16, seed=seed + COUNTER)
    if USE_ELSE_NOISE:
        COUNTER += 1
    timesteps = get_schedule(
        num_steps,
        (width // 8) * (height // 8) // (16 * 16),
        shift=True,
    )

    # ------ paint ------ #
    x_0 = None
    x_1 = None
    if warped_x is not None and mask is not None:
        x_0 = x
        x_1 = torch.from_numpy(warped_x).permute(2, 0,
                                                 1).unsqueeze(0).to(self.device).to(torch.bfloat16)
        x_1 = x_1 / 127.5 - 1
        x_1 = self.ae.encode(x_1)

        mask = cv2.resize(mask.astype(np.float32), (x_0.shape[-1], x_0.shape[-2]),
                          interpolation=cv2.INTER_NEAREST)
        mask = mask[:, :, np.newaxis]
        mask = torch.from_numpy(mask).permute(2, 0, 1)[None]
        mask = mask.expand(-1, 16, -1, -1).to(self.device).to(torch.bfloat16)
    # ------ paint ------ #

    torch.manual_seed(seed)
    with torch.no_grad():
        # if self.offload:
        #     self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
        # self.t5 = self.t5.to(self.device)
        inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)
        neg_inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt)

        # if self.offload:
        #     self.offload_model_to_cpu(self.t5, self.clip)
        #     self.model = self.model.to(self.device)
        if self.controlnet_loaded:
            x = denoise_controlnet(
                self,
                x_0,
                x_1,
                mask,
                prompt,
                height,
                width,
                self.model,
                **inp_cond,
                controlnet=self.controlnet,
                timesteps=timesteps,
                guidance=guidance,
                controlnet_cond=controlnet_image,
                timestep_to_start_cfg=timestep_to_start_cfg,
                neg_txt=neg_inp_cond['txt'],
                neg_txt_ids=neg_inp_cond['txt_ids'],
                neg_vec=neg_inp_cond['vec'],
                true_gs=true_gs,
                controlnet_gs=control_weight,
                image_proj=image_proj,
                neg_image_proj=neg_image_proj,
                ip_scale=ip_scale,
                neg_ip_scale=neg_ip_scale,
                num_replacement=NUM_REPLACEMENT,
            )
        else:
            x = denoise(
                self.model,
                **inp_cond,
                timesteps=timesteps,
                guidance=guidance,
                timestep_to_start_cfg=timestep_to_start_cfg,
                neg_txt=neg_inp_cond['txt'],
                neg_txt_ids=neg_inp_cond['txt_ids'],
                neg_vec=neg_inp_cond['vec'],
                true_gs=true_gs,
                image_proj=image_proj,
                neg_image_proj=neg_image_proj,
                ip_scale=ip_scale,
                neg_ip_scale=neg_ip_scale,
            )

        # if self.offload:
        #     self.offload_model_to_cpu(self.model)
        #     self.ae.decoder.to(x.device)
        x = unpack(x, height, width)
        x = self.ae.decode(x)
        # self.offload_model_to_cpu(self.ae.decoder)

    x1 = x.clamp(-1, 1).float()
    x1 = rearrange(x1, "b c h w -> b h w c")
    x1 = (127.5 * (x1 + 1.0)).cpu().byte().numpy()
    output_imgs = [Image.fromarray(x) for x in x1]
    return output_imgs


def call(
    self,
    warped_x,
    mask,
    prompt: str,
    image_prompt: Image = None,
    controlnet_image: Image = None,
    width: int = 512,
    height: int = 512,
    guidance: float = 4,
    num_steps: int = 50,
    seed: int = 123456789,
    true_gs: float = 3,
    control_weight: float = 0.9,
    ip_scale: float = 1.0,
    neg_ip_scale: float = 1.0,
    neg_prompt: str = '',
    neg_image_prompt: Image = None,
    timestep_to_start_cfg: int = 0,
    itself=False,
):
    width = 16 * (width // 16)
    height = 16 * (height // 16)
    image_proj = None
    neg_image_proj = None
    if not (image_prompt is None and neg_image_prompt is None):
        assert self.ip_loaded, 'You must setup IP-Adapter to add image prompt as input'

        if image_prompt is None:
            image_prompt = np.zeros((width, height, 3), dtype=np.uint8)
        if neg_image_prompt is None:
            neg_image_prompt = np.zeros((width, height, 3), dtype=np.uint8)

        image_proj = self.get_image_proj(image_prompt)
        neg_image_proj = self.get_image_proj(neg_image_prompt)

    if self.controlnet_loaded:
        if itself:
            controlnet_image = controlnet_image.resize((width, height))
            controlnet_image = np.array(controlnet_image)
        else:
            controlnet_image = self.annotator(controlnet_image, width, height)
        controlnet_image = torch.from_numpy((np.array(controlnet_image) / 127.5) - 1)
        controlnet_image = controlnet_image.permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16).to(
            self.device)

    return self.forward(
        warped_x,
        mask,
        prompt,
        width,
        height,
        guidance,
        num_steps,
        seed,
        controlnet_image,
        timestep_to_start_cfg=timestep_to_start_cfg,
        true_gs=true_gs,
        control_weight=control_weight,
        neg_prompt=neg_prompt,
        image_proj=image_proj,
        neg_image_proj=neg_image_proj,
        ip_scale=ip_scale,
        neg_ip_scale=neg_ip_scale,
    )


def create_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--neg_prompt", type=str, default="", help="The input text negative prompt")
    parser.add_argument("--img_prompt", type=str, default=None, help="Path to input image prompt")
    parser.add_argument("--neg_img_prompt",
                        type=str,
                        default=None,
                        help="Path to input negative image prompt")
    parser.add_argument("--ip_scale",
                        type=float,
                        default=1.0,
                        help="Strength of input image prompt")
    parser.add_argument("--neg_ip_scale",
                        type=float,
                        default=1.0,
                        help="Strength of negative input image prompt")
    parser.add_argument("--local_path",
                        type=str,
                        default=None,
                        help="Local path to the model checkpoint (Controlnet)")
    parser.add_argument("--repo_id",
                        type=str,
                        default=None,
                        help="A HuggingFace repo id to download model (Controlnet)")
    parser.add_argument("--name",
                        type=str,
                        default=None,
                        help="A filename to download from HuggingFace")
    parser.add_argument("--ip_repo_id",
                        type=str,
                        default=None,
                        help="A HuggingFace repo id to download model (IP-Adapter)")
    parser.add_argument("--ip_name",
                        type=str,
                        default=None,
                        help="A IP-Adapter filename to download from HuggingFace")
    parser.add_argument("--ip_local_path",
                        type=str,
                        default=None,
                        help="Local path to the model checkpoint (IP-Adapter)")
    parser.add_argument("--lora_repo_id",
                        type=str,
                        default=None,
                        help="A HuggingFace repo id to download model (LoRA)")
    parser.add_argument("--lora_name",
                        type=str,
                        default=None,
                        help="A LoRA filename to download from HuggingFace")
    parser.add_argument("--lora_local_path",
                        type=str,
                        default=None,
                        help="Local path to the model checkpoint (Controlnet)")
    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")
    parser.add_argument("--offload",
                        action='store_true',
                        help="Offload model to CPU when not in use")
    parser.add_argument("--itself", action='store_true', help="use annotation itself")
    parser.add_argument("--use_ip", action='store_true', help="Load IP model")
    parser.add_argument("--use_lora", action='store_true', help="Load Lora model")
    parser.add_argument("--use_controlnet", action='store_true', help="Load Controlnet model")
    parser.add_argument("--num_images_per_prompt",
                        type=int,
                        default=1,
                        help="The number of images to generate per prompt")
    parser.add_argument("--image", type=str, default=None, help="Path to image")
    parser.add_argument("--lora_weight",
                        type=float,
                        default=1.0,
                        help="Lora model strength (from 0 to 1.0)")
    parser.add_argument("--control_weight",
                        type=float,
                        default=0.8,
                        help="Controlnet model strength (from 0 to 1.0)")
    parser.add_argument("--model_type",
                        type=str,
                        default="flux-dev",
                        choices=("flux-dev", "flux-dev-fp8", "flux-schnell"),
                        help="Model type to use (flux-dev, flux-dev-fp8, flux-schnell)")
    parser.add_argument("--width", type=int, default=1024, help="The width for generated image")
    parser.add_argument("--height", type=int, default=1024, help="The height for generated image")
    parser.add_argument("--guidance",
                        type=float,
                        default=4,
                        help="The guidance for diffusion process")
    parser.add_argument("--true_gs", type=float, default=3.5, help="true guidance")
    parser.add_argument("--timestep_to_start_cfg",
                        type=int,
                        default=5,
                        help="timestep to start true guidance")
    parser.add_argument("--save_path", type=str, default='results', help="Path to save")

    # hacked ------------------------------------------------------------------------------#
    parser.add_argument("--control_type", type=str, default="hed", choices=("canny", "hed"))
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--epoch", type=int, default=400, help="Path to save")
    parser.add_argument("--target", type=str, default='assets/sampleZ02_tmp1')
    parser.add_argument("--prompt", type=str, nargs='+', default=[''], help="The input text prompt")
    parser.add_argument("--use_else_noise", action='store_true')
    parser.add_argument("--num_replacement", type=int, default=14)

    # paint
    if gethostname() == "comar-System-Product-Name":
        parser.add_argument("--pfix", type=str, default='spatown3')
        parser.add_argument("--paint", action='store_true', default=True)
        parser.add_argument("--num_steps", type=int, default=2)
    else:
        parser.add_argument("--pfix", type=str, default='default')
        parser.add_argument("--paint", action='store_true')
        parser.add_argument("--num_steps", type=int, default=25)
    # -------------------------------------------------------------------------------------#

    return parser


def update_model_with_lora(model, checkpoint, lora_weight, device):
    rank = get_lora_rank(checkpoint)
    lora_attn_procs = {}

    for name, _ in model.attn_processors.items():
        lora_state_dict = {}
        for k in checkpoint.keys():
            if name in k:
                lora_state_dict[k[len(name) + 1:]] = checkpoint[k] * lora_weight

        if len(lora_state_dict):
            if name.startswith("single_blocks"):
                lora_attn_procs[name] = SingleStreamBlockLoraProcessor(dim=3072, rank=rank)
            else:
                lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(dim=3072, rank=rank)
            lora_attn_procs[name].load_state_dict(lora_state_dict)
            lora_attn_procs[name].to(device)
        else:
            if name.startswith("single_blocks"):
                lora_attn_procs[name] = SingleStreamBlockProcessor()
            else:
                lora_attn_procs[name] = DoubleStreamBlockProcessor()

    model.set_attn_processor(lora_attn_procs)


def main(args):
    # get corres end
    if hasattr(args, 'corres_end') and args.corres_end is not None:
        #corres_end = [Image.open(item) for item in args.corres_end]
        corres_end = [cv2.imread(item, cv2.IMREAD_UNCHANGED) for item in args.corres_end]
        corres_end.insert(0, None)
    else:
        corres_end = None

    # get hed edge
    if args.image:
        image = [Image.open(item).convert('RGB') for item in args.image]
    else:
        image = None

    xflux_pipeline = XFluxPipeline(args.model_type, args.device, args.offload)
    if args.use_ip:
        print('load ip-adapter:', args.ip_local_path, args.ip_repo_id, args.ip_name)
        xflux_pipeline.set_ip(args.ip_local_path, args.ip_repo_id, args.ip_name)
    if args.use_controlnet:
        print('load controlnet:', args.local_path, args.repo_id, args.name)
        xflux_pipeline.set_controlnet(args.control_type, args.local_path, args.repo_id, args.name)
    if args.use_lora:
        # ------------------------------------------------------------------------------------------- #
        checkpoint = load_checkpoint(args.lora_local_path, None, None)
        checkpoint_controlnet = load_checkpoint(args.control_lora_local_path, None, None)
        update_model_with_lora(xflux_pipeline.model, checkpoint, args.lora_weight,
                               xflux_pipeline.device)
        update_model_with_lora(xflux_pipeline.controlnet, checkpoint_controlnet, args.lora_weight,
                               xflux_pipeline.device)
        # ------------------------------------------------------------------------------------------- #

    image_prompt = Image.open(args.img_prompt) if args.img_prompt else None
    neg_image_prompt = Image.open(args.neg_img_prompt) if args.neg_img_prompt else None

    xflux_pipeline.__class__ = type("XFluxPipeline", (xflux_pipeline.__class__, ),
                                    {"__call__": call})
    xflux_pipeline.forward = MethodType(forward, xflux_pipeline)

    result = None
    for i, (img, prompt) in enumerate(zip(image, args.prompt)):
        if args.paint and result is not None:
            warped_x, mask, = Warp(np.array(result), np.array(corres_end[i]))
        else:
            warped_x = mask = None

        result = xflux_pipeline(
            # inpaint
            warped_x=warped_x,
            mask=mask,
            # inpaint
            prompt=prompt,
            controlnet_image=img,
            width=args.width,
            height=args.height,
            guidance=args.guidance,
            num_steps=args.num_steps,
            seed=args.seed,
            true_gs=args.true_gs,
            control_weight=args.control_weight,
            neg_prompt=args.neg_prompt,
            timestep_to_start_cfg=args.timestep_to_start_cfg,
            image_prompt=image_prompt,
            neg_image_prompt=neg_image_prompt,
            ip_scale=args.ip_scale,
            neg_ip_scale=args.neg_ip_scale,
            itself=args.itself,
        )[0]

        path = os.path.join(args.save_path, f"image_{i:02d}.png")
        result.save(path)
        # ------ paint ------#
        if warped_x is not None:
            Image.fromarray(warped_x.astype(np.uint8)).save(
                os.path.join(args.save_path, f"warped_x_{i:02d}.png"))
        if mask is not None:
            Image.fromarray((mask[..., 0] * 255).astype(np.uint8),
                            mode='L').save(os.path.join(args.save_path, f"mask_{i:02d}.png"))
        # ------ paint ------#


# -------------------- get binary mask start -------------------- #


def colorcode2correspondence(colorcode, bit=16):
    height, width, _ = colorcode.shape
    fn = get_coord_mapper(width=width, height=height, bit=bit)
    r, g, b = colorcode[..., 2], colorcode[..., 1], colorcode[..., 0]
    c_x, c_y, mask = fn(r, g, b)
    x_src, y_src = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')

    x_1 = c_x[mask]
    y_1 = c_y[mask]
    x_2 = x_src[mask]
    y_2 = y_src[mask]

    return x_1, y_1, x_2, y_2


def get_coord_mapper(width, height, bit):
    val_max = 2**bit - 1

    def fn(r, g, b):
        # b 값이 100 미만인 경우 (-1, -1)

        mask = b >= 100
        g = val_max - g
        x = np.where(mask, r / (val_max + 1) * width, -10000).astype('int32')
        y = np.where(mask, g / (val_max + 1) * height, -10000).astype('int32')
        return x, y, mask

    return fn


def resize_colorcode(colorcode, width, height):
    colorcode = cv2.resize(colorcode, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = (colorcode[..., 0] == (2**16 - 1))[..., None]
    colorcode = colorcode * mask
    return colorcode


def Warp(x, c, mask_dilation=(5, 8 + 4)):
    height, width, _ = x.shape

    c = resize_colorcode(c, width, height)
    c = np.clip(c, 0, 65535).astype(np.uint16)

    mask = ((c[..., 0] == 0) * 1).astype('uint16')
    if mask_dilation is not None:
        kernel = np.ones((mask_dilation[0], mask_dilation[0]), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=mask_dilation[1])[..., None]
        mask = cv2.GaussianBlur(mask, (9, 9), 3)[..., None]

    x_w = np.zeros_like(x)
    x_1, y_1, x_2, y_2 = colorcode2correspondence(c)
    x_w[y_2, x_2] = x[y_1, x_1]

    return x_w, mask


# -------------------- get binary mask end -------------------- #

if __name__ == "__main__":
    args = create_argparser().parse_args()

    NUM_REPLACEMENT = args.num_replacement
    USE_ELSE_NOISE = args.use_else_noise

    name = Path(os.path.basename(__file__)).stem
    epoch = args.epoch

    paths = sorted(glob(join(args.target, 'arc*')))
    num_frame = len(paths) + 1

    # get correspond end path
    if args.paint:
        print(f'Paint keyframes...')
        corres_end_path = []
        for i, path in enumerate(paths):
            # get corres end
            corres_end_path.append(glob(join(path, 'corres_end.png'))[0])
        args.corres_end = corres_end_path

    # get hed edge path
    image = []
    for i, path in enumerate(paths):
        path_heds = sorted(glob(join(path, 'hed/*.png')))
        if i == 0:
            image.append(path_heds[0])
            image.append(path_heds[-1])
        else:
            image.append(path_heds[-1])
    args.image = image

    if args.seed == -1:
        args.seed = random.randint(0, 100000)

    args.itself = True

    args.use_controlnet = True
    assert args.control_type == "hed"
    args.repo_id = "XLabs-AI/flux-controlnet-hed-v3"
    args.name = "flux-hed-controlnet-v3.safetensors"

    if len(args.prompt) < num_frame:
        num_remain = num_frame - len(args.prompt)
        args.prompt = args.prompt + [args.prompt[-1]] * num_remain

    hash_func = hashlib.new('sha256')
    hash_func.update(''.join(args.prompt).encode('utf-8'))
    hc = hash_func.hexdigest()

    path_dir = f"{name[:4]}_{args.pfix}_p{args.prompt[0].replace(' ', '_')}-{hc[:8]}_e{args.epoch:03d}_s{args.seed:06d}_r{args.num_replacement:02d}_ip{int(args.paint)}"

    path_log = join(args.target, 'multiviews', path_dir)
    os.makedirs(path_log, exist_ok=True)

    args.width, args.height = Image.open(path_heds[0]).size

    args.model_type = "flux-dev"
    args.timestep_to_start_cfg = 1
    args.save_path = path_log
    args.lora_weight = 1.0
    args.use_lora = True
    args.lora_local_path = f'logs/{name[:-2]}00/{args.pfix}/checkpoint-{epoch}/lora.safetensors'
    args.control_lora_local_path = f'logs/{name[:-2]}00/{args.pfix}/checkpoint-{epoch}/controlnet.lora.safetensors'

    if gethostname() == "comar-System-Product-Name":
        args.offload = True

    main(args)

