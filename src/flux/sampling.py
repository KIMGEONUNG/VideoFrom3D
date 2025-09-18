import math
from typing import Callable
from socket import gethostname

import torch
from einops import rearrange, repeat
from torch import Tensor
from tqdm import tqdm

from .model import Flux
from .modules.conditioner import HFEmbedder


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor,
            prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1)**sigma)


def get_lin_function(x1: float = 256,
                     y1: float = 0.5,
                     x2: float = 4096,
                     y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
        model: Flux,
        # model input
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        vec: Tensor,
        neg_txt: Tensor,
        neg_txt_ids: Tensor,
        neg_vec: Tensor,
        # sampling parameters
        timesteps: list[float],
        guidance: float = 4.0,
        true_gs=1,
        timestep_to_start_cfg=0,
        # ip-adapter parameters
        image_proj: Tensor = None,
        neg_image_proj: Tensor = None,
        ip_scale: Tensor | float = 1.0,
        neg_ip_scale: Tensor | float = 1.0):
    i = 0
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0], ), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in tqdm(list(zip(timesteps[:-1], timesteps[1:]))):

        # Hacked -----------------------------------#
        for m in model.double_blocks:
            m.timestamp = i
        for m in model.single_blocks:
            m.timestamp = i
        # ------------------------------------------#

        t_vec = torch.full((img.shape[0], ), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            image_proj=image_proj,
            ip_scale=ip_scale,
        )
        if i >= timestep_to_start_cfg:
            neg_pred = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                image_proj=neg_image_proj,
                ip_scale=neg_ip_scale,
            )
            pred = neg_pred + true_gs * (pred - neg_pred)
        img = img + (t_prev - t_curr) * pred
        i += 1
    return img


def denoise_sdedit(
        self,
        model: Flux,
        # model input
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        vec: Tensor,
        neg_txt: Tensor,
        neg_txt_ids: Tensor,
        neg_vec: Tensor,
        # sampling parameters
        timesteps: list[float],
        guidance: float = 4.0,
        true_gs=1,
        timestep_to_start_cfg=0,
        # ip-adapter parameters
        image_proj: Tensor = None,
        neg_image_proj: Tensor = None,
        ip_scale: Tensor | float = 1.0,
        neg_ip_scale: Tensor | float = 1.0,
        num_replacement = 10,
        x_0=None,
        x_1=None,
        mask=None,
        height=None,
        width=None,
        prompt=None,
        ):
    i = 0
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0], ), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in tqdm(list(zip(timesteps[:-1], timesteps[1:]))):

        # Hacked -----------------------------------#
        for m in model.double_blocks:
            m.timestamp = i
        for m in model.single_blocks:
            m.timestamp = i
        # ------------------------------------------#

        if num_replacement > i:
            # unpack
            if i == 0:
                z_t_new = x_0
            else:
                z_t = unpack(img, height, width)
                x_t = (1 - t_curr) * x_1 + t_curr * x_0
                z_t_new = mask * z_t + (1 - mask) * x_t

            # pack
            img = prepare(t5=self.t5, clip=self.clip, img=z_t_new, prompt=prompt)['img']

        t_vec = torch.full((img.shape[0], ), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            image_proj=image_proj,
            ip_scale=ip_scale,
        )
        if i >= timestep_to_start_cfg:
            neg_pred = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                image_proj=neg_image_proj,
                ip_scale=neg_ip_scale,
            )
            pred = neg_pred + true_gs * (pred - neg_pred)
        img = img + (t_prev - t_curr) * pred
        i += 1
    return img


def denoise_controlnet(
    # ------ paint ------ #
    self,
    x_0,
    x_1,
    mask,
    prompt,
    height,
    width,
    # ------ paint ------ #
    model: Flux,
    controlnet: None,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    controlnet_cond,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs=1,
    controlnet_gs=0.7,
    timestep_to_start_cfg=0,
    # ip-adapter parameters
    image_proj: Tensor = None,
    neg_image_proj: Tensor = None,
    ip_scale: Tensor | float = 1,
    neg_ip_scale: Tensor | float = 1,
    num_replacement = None,
    controlrange=list(range(25)),
):

    paint = True if x_0 is not None and x_1 is not None else False

    # this is ignored for schnell
    i = 0
    guidance_vec = torch.full((img.shape[0], ), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in tqdm(list(zip(timesteps[:-1], timesteps[1:]))):

        # Hacked -----------------------------------#
        for m in model.double_blocks:
            m.timestamp = i
        for m in model.single_blocks:
            m.timestamp = i
        # ------------------------------------------#

        # ------ paint ------ #
        if paint and num_replacement > i:
            # unpack
            if i == 0:
                z_t_new = x_0
            else:
                z_t = unpack(img, height, width)
                x_t = (1 - t_curr) * x_1 + t_curr * x_0
                z_t_new = mask * z_t + (1 - mask) * x_t

            # pack
            img = prepare(t5=self.t5, clip=self.clip, img=z_t_new, prompt=prompt)['img']
        # ------ paint ------ #

        t_vec = torch.full((img.shape[0], ), t_curr, dtype=img.dtype, device=img.device)

        if controlrange is not None and i in controlrange:
            block_res_samples = controlnet(
                img=img,
                img_ids=img_ids,
                controlnet_cond=controlnet_cond,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
            )
        else:
            block_res_samples = None

        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            block_controlnet_hidden_states=[i * controlnet_gs for i in block_res_samples] if block_res_samples else None,
            image_proj=image_proj,
            ip_scale=ip_scale,
        )
        if i >= timestep_to_start_cfg:
            if controlrange is not None and i in controlrange:
                neg_block_res_samples = controlnet(
                    img=img,
                    img_ids=img_ids,
                    controlnet_cond=controlnet_cond,
                    txt=neg_txt,
                    txt_ids=neg_txt_ids,
                    y=neg_vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )
            else:
                neg_block_res_samples = None
            neg_pred = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                block_controlnet_hidden_states=[i * controlnet_gs for i in neg_block_res_samples] if neg_block_res_samples else None,
                image_proj=neg_image_proj,
                ip_scale=neg_ip_scale,
            )
            pred = neg_pred + true_gs * (pred - neg_pred)

        img = img + (t_prev - t_curr) * pred
        i += 1

        if gethostname() == "comar-System-Product-Name":
            from torchvision.transforms import ToPILImage
            ToPILImage()(self.ae.decode(unpack((img - t_curr * pred)/(1- t_curr), height, width))[0].clamp(-1,1).float() * 0.5 + 0.5).save(f'{i:04d}.png')

    return img


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
