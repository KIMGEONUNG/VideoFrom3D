<div align="center">

<h1>
    📽️ VideoFrom3D 📽️<br> 
     <sub>3D Scene Video Generation via Complementary Image and Video Diffusion Models</sub>
</h1>

<div>
    <a href='https://kimgeonung.github.io/' target='_blank'>Geonung Kim</a>&emsp;
    <a target='_blank'>Janghyeok Han</a>&emsp;
    <a href='https://www.scho.pe.kr/' target='_blank'>Sunghyun Cho</a>&emsp;
</div>
<div>
    POSTECH CG Lab.
</div>

<div>
    <strong>SIGGRAPH-ASIA 2025 Conference </strong>
</div>

<div>
    <h4 align="center">
        <a href="https://kimgeonung.github.io/VideoFrom3D/" target='_blank'>
        <img src="https://img.shields.io/badge/🐳-Project%20Page-blue">
        </a>
        <!-- <a href="https://arxiv.org/abs/2401.17547" target='_blank'> -->
        <img src="https://img.shields.io/badge/arXiv-2401.17547-b31b1b.svg">
        </a>
    </h4>
</div>

![teaser](assets-readme/teaser.jpg) 
---

</div>

## 🔥 Update

- [2025.09.08] The repository is created.

## 🔧 Todo

- [x] Release model checkpoint
- [x] Release inference code
- [ ] Release preprocessing code 

## ⌨️ Inference

### Install Environment

```sh

# create conda environment
conda create --name videofrom3d python=3.10

# install pytorch (use propor cuda version option)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# install other packages
pip install -r requirements.txt
 
``` 

### Sparse Appearance-guided Sampling (SAG)

#### Distribution Alignment

For the below example, the trained loras are saved in `./loras`

```sh

# Options
# --path_image : reference image path(s)
# --pfix       : name of LoRA to be saved 
# --prompt     : uninque identifier prompts for each reference image

# For a single style
accelerate launch --num_processes 1 --main_process_port=4401 sag_distribution_alignment.py \
    --path_image assets/references/spatown.png \
    --pfix spatown

# For multiple styles for each identifier prompt
accelerate launch --num_processes 1 --main_process_port=4401 sag_distribution_alignment.py \
    --path_image assets/references/exterior.png assets/references/interior.png \
    --prompt 'exterior' 'interior' \
    --pfix school
```

#### Anchor View Generation

For the below example, we provide `assets/sampleA` as an example. The generated anchor views are saved in `./assets/sampleA/multiviews/spatown_p-e3b0c442_e400_s075157_r12_ip1
`
```sh
# Options
# --pfix            : target lora name
# --epoch           : target lora epoch
# --target          : target input
# --num_replacement : the number of replacements for sparse appearance (warped image)
# --prompt          : additional prompt for style varation
# --offload         : use lower memory

python sag_generate_anchor_view.py --epoch 400 --target assets/sampleA --pfix spatown --num_replacement 12

```

###  Geometry-guided Generative Inbetweening (GGI)

Before started, you first download the checkpoint-1350 in [here](https://drive.google.com/drive/folders/1IhI9qDv6tH5T7XzeEjx27UYw2EqZ7MKY?usp=sharing), and move it in `./checkpoints`, e.g. , `./checkpoints/checkpoint-1350`. 
The generated video sequence is saved in `assets/sampleA/multiviews/spatown_p-e3b0c442_e400_s051106_r12_ip1/d0.5_e1350_n30`

```sh
# Options
# --target  : target anchor view path
# --offload : use lower memory

python ggi.py --target assets/sampleA/multiviews/spatown_p-e3b0c442_e400_s051106_r12_ip1
```

## ☕️ Acknowledgment

- We borrowed the readme format from [Upscale-A-Video](https://github.com/sczhou/Upscale-A-Video) 
- We finetune a pretrained video diffusion model, [CogVideoX](https://github.com/zai-org/CogVideo) 
- We generate anchor views using [Flux ControlNet](https://huggingface.co/XLabs-AI/flux-controlnet-collections) 


## 📄 Citation

```
Comming Soon
<!-- @inproceedings{Kim2022BigColor, -->
<!--   title     = {BigColor: Colorization using a Generative Color Prior for Natural Images}, -->
<!--   author    = {Geonung Kim,Kyoungkook Kang,Seongtae Kim,Hwayoon Lee,Sehoon Kim,Jonghyun Kim,Seung-Hwan Baek,Sunghyun Cho}, -->
<!--   booktitle = {European Conference on Computer Vision (ECCV)}, -->
<!--   year      = {2022} -->
<!-- } -->

```
