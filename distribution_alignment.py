import argparse
from socket import gethostname
from os.path import join
from PIL import Image
import logging
import math
import os
import re
import random
import shutil
from pathlib import Path
from safetensors.torch import save_file

import datasets
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from copy import deepcopy
import diffusers
from diffusers.optimization import get_scheduler
from einops import rearrange
import sys
from accelerate import cpu_offload

from controlnet_aux import HEDdetector

sys.path.append('.')
from src.flux.sampling import get_schedule, prepare
from src.flux.util import (
    load_ae,
    load_clip,
    load_flow_model2,
    load_t5,
    load_controlnet,
    load_checkpoint,
)
from src.flux.modules.layers import DoubleStreamBlockLoraProcessor, SingleStreamBlockLoraProcessor
from src.flux.xflux_pipeline import XFluxSampler

from image_datasets.dataset import loader_custom

logger = get_logger(__name__, log_level="INFO")


def get_models(name: str, device, offload: bool, is_schnell: bool):
    t5 = load_t5('cpu', max_length=256 if is_schnell else 512)
    clip = load_clip('cpu')
    clip.requires_grad_(False)
    vae = load_ae(name, device="cpu")
    controlnet = load_controlnet('flux-dev', 'cpu')
    controlnet.load_state_dict(
        load_checkpoint(
            None,
            "XLabs-AI/flux-controlnet-hed-v3",
            "flux-hed-controlnet-v3.safetensors",
        ),
        strict=False,
    )

    # cpu_offload(t5, device)
    t5.to(device)
    # cpu_offload(clip, device)
    clip.to(device)
    cpu_offload(vae, device)
    model = load_flow_model2(name, device="cpu")
    return model, vae, t5, clip, controlnet


def parse():
    p = argparse.ArgumentParser()
    if gethostname() == "comar-System-Product-Name":
        p.add_argument('--path_image', type=str, nargs='+', default=['datasets/exterior.png'])
    else:
        p.add_argument('--path_image', type=str, nargs='+', required=True)
    p.add_argument('--prompt', type=str, nargs='+', default=None)
    p.add_argument('--pfix', default='default-')
    return p.parse_args()


def main():

    args = parse()
    # Set basic
    name = Path(os.path.basename(__file__)).stem
    logging_dir = f'logs/{name}/{args.pfix}'
    os.makedirs(logging_dir, exist_ok=False)

    path_image = args.path_image
    prompt = args.prompt
    args = OmegaConf.load(f'configs/{name}.yaml')

    for i, image in enumerate(path_image):
        Image.open(image).save(join(logging_dir, f'image{i:02d}.png'))

    if prompt is None:
        prompt = [''] * len(path_image)

    assert len(path_image) == len(prompt)

    args.data_config.img_dir = path_image
    args.data_config.prompt = prompt
    args.output_dir = logging_dir

    is_schnell = args.model_name == "flux-schnell"
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir,
                                                      logging_dir=logging_dir)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    dit, vae, t5, clip, controlnet = get_models(name=args.model_name,
                                                device=accelerator.device,
                                                offload=False,
                                                is_schnell=is_schnell)
    vae.requires_grad_(False)
    t5.requires_grad_(False)
    clip.requires_grad_(False)

    lora_attn_procs = {}

    if args.double_blocks is None:
        double_blocks_idx = list(range(19))
    else:
        double_blocks_idx = [int(idx) for idx in args.double_blocks.split(",")]

    if args.single_blocks is None:
        single_blocks_idx = list(range(38))
    elif args.single_blocks is not None:
        single_blocks_idx = [int(idx) for idx in args.single_blocks.split(",")]

    for name, attn_processor in dit.attn_processors.items():
        match = re.search(r'\.(\d+)\.', name)
        if match:
            layer_index = int(match.group(1))

        if name.startswith("double_blocks") and layer_index in double_blocks_idx:
            print("setting LoRA Processor for", name)
            lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(dim=3072, rank=args.rank)
        elif name.startswith("single_blocks") and layer_index in single_blocks_idx:
            print("setting LoRA Processor for", name)
            lora_attn_procs[name] = SingleStreamBlockLoraProcessor(dim=3072, rank=args.rank)
        else:
            lora_attn_procs[name] = attn_processor

    dit.set_attn_processor(lora_attn_procs)
    dit = dit.to(weight_dtype)
    dit.train()
    optimizer_cls = torch.optim.AdamW
    for n, param in dit.named_parameters():
        if '_lora' not in n:
            param.requires_grad = False
        else:
            print(n)

    # HACK -------------------------------------------------------------------------------------- #
    lora_attn_procs = {}

    if args.double_blocks is None:
        double_blocks_idx = list(range(19))
    else:
        double_blocks_idx = [int(idx) for idx in args.double_blocks.split(",")]

    if args.single_blocks is None:
        single_blocks_idx = list(range(38))
    elif args.single_blocks is not None:
        single_blocks_idx = [int(idx) for idx in args.single_blocks.split(",")]

    for name, attn_processor in controlnet.attn_processors.items():
        match = re.search(r'\.(\d+)\.', name)
        if match:
            layer_index = int(match.group(1))

        if name.startswith("double_blocks") and layer_index in double_blocks_idx:
            print("setting LoRA Processor for", name)
            lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(dim=3072, rank=args.rank)
        elif name.startswith("single_blocks") and layer_index in single_blocks_idx:
            print("setting LoRA Processor for", name)
            lora_attn_procs[name] = SingleStreamBlockLoraProcessor(dim=3072, rank=args.rank)
        else:
            lora_attn_procs[name] = attn_processor

    controlnet.set_attn_processor(lora_attn_procs)
    controlnet = controlnet.to(weight_dtype)
    # controlnet = controlnet.to(weight_dtype)
    controlnet.train()
    optimizer_cls = torch.optim.AdamW
    for n, param in controlnet.named_parameters():
        if '_lora' not in n:
            param.requires_grad = False
        else:
            print(n)
    # ------------------------------------------------------------------------------------------- #

    optimizer = optimizer_cls(
        [p for p in dit.parameters() if p.requires_grad] +
        [p for p in controlnet.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataloader = loader_custom(
        aux=HEDdetector.from_pretrained("lllyasviel/Annotators"),
        **args.data_config,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    global_step = 0
    first_epoch = 0

    dit, controlnet, optimizer, _, lr_scheduler = accelerator.prepare(dit, controlnet, optimizer,
                                                                      deepcopy(train_dataloader),
                                                                      lr_scheduler)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    timesteps = get_schedule(
        999,
        (1024 // 8) * (1024 // 8) // 4,
        shift=True,
    )
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(dit, controlnet):
                img, prompts, controlnet_cond = batch
                # __import__('pdb').set_trace()
                controlnet_cond = controlnet_cond.to(accelerator.device)
                with torch.no_grad():
                    x_1 = vae.encode(img.to(accelerator.device).to(torch.float32))
                    inp = prepare(t5=t5, clip=clip, img=x_1, prompt=prompts)
                    x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

                bs = img.shape[0]
                t = torch.tensor([timesteps[random.randint(0, 999)]]).to(accelerator.device)
                x_0 = torch.randn_like(x_1).to(accelerator.device)
                x_t = (1 - t) * x_1 + t * x_0
                bsz = x_1.shape[0]
                guidance_vec = torch.full((x_t.shape[0], ), 1, device=x_t.device, dtype=x_t.dtype)

                # Predict the noise residual and compute loss
                block_res_samples = controlnet(
                    img=x_t.to(weight_dtype),
                    img_ids=inp['img_ids'].to(weight_dtype),
                    txt=inp['txt'].to(weight_dtype),
                    controlnet_cond=controlnet_cond.to(weight_dtype),
                    txt_ids=inp['txt_ids'].to(weight_dtype),
                    y=inp['vec'].to(weight_dtype),
                    timesteps=t.to(weight_dtype),
                    guidance=guidance_vec.to(weight_dtype),
                )
                model_pred = dit(
                    img=x_t.to(weight_dtype),
                    img_ids=inp['img_ids'].to(weight_dtype),
                    txt=inp['txt'].to(weight_dtype),
                    txt_ids=inp['txt_ids'].to(weight_dtype),
                    y=inp['vec'].to(weight_dtype),
                    timesteps=t.to(weight_dtype),
                    guidance=guidance_vec.to(weight_dtype),
                    block_controlnet_hidden_states=block_res_samples,
                )

                loss = F.mse_loss(model_pred.float(), (x_0 - x_1).float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(dit.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if not args.disable_sampling and global_step % args.sample_every == 0:
                    if accelerator.is_main_process:
                        print(f"Sampling images for step {global_step}...")
                        sampler = XFluxSampler(clip=clip,
                                               t5=t5,
                                               ae=vae,
                                               model=dit,
                                               device=accelerator.device)
                        images = []
                        for i, prompt in enumerate(args.sample_prompts):
                            result = sampler(prompt=prompt,
                                             width=args.sample_width,
                                             height=args.sample_height,
                                             num_steps=args.sample_steps)
                            print(f"Result for prompt #{i} is generated")

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)

                    # accelerator.save_state(save_path)
                    unwrapped_model_state = accelerator.unwrap_model(dit).state_dict()

                    # save checkpoint in safetensors format
                    lora_state_dict = {
                        k: unwrapped_model_state[k]
                        for k in unwrapped_model_state.keys() if '_lora' in k
                    }
                    save_file(lora_state_dict, os.path.join(save_path, "lora.safetensors"))

                    # HACK ---------------------------------------------------------------------- #
                    # accelerator.save_state(save_path)
                    unwrapped_model_state = accelerator.unwrap_model(controlnet).state_dict()

                    # save checkpoint in safetensors format
                    lora_state_dict = {
                        k: unwrapped_model_state[k]
                        for k in unwrapped_model_state.keys() if '_lora' in k
                    }
                    save_file(lora_state_dict, os.path.join(save_path,
                                                            "controlnet.lora.safetensors"))
                    # --------------------------------------------------------------------------- #

                    logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        progress_bar.update(1)
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
