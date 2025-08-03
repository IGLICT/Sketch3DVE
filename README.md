## ***Sketch3DVE: Sketch-based 3D-Aware Scene Video Editing***
<!-- ![](./assets/logo_long.png#gh-light-mode-only){: width="50%"} -->
<!-- ![](./assets/logo_long_dark.png#gh-dark-mode-only=100x20) -->
<div align="center">
<img src='assets/logo.png' style="height:100px"></img>

<a href='https://arxiv.org/abs/2503.23284'><img src='https://img.shields.io/badge/arXiv-2405.17933-b31b1b.svg'></a> &nbsp;
<a href='http://geometrylearning.com/SketchVideo/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<a href='https://www.youtube.com/watch?v=ABnx9tUvI2M'><img src='https://img.shields.io/badge/Youtube-Video-b31b1b.svg'></a><br>

<strong> SIGGRAPH 2025</strong>


</div>
 
## &#x1F680; Introduction

We propose Sketch3DVE, a sketch-based 3D-aware video editing method to enable detailed local manipulation of videos with significant viewpoint changes.  Please check our project page and paper for more information. <br>


### 1. 3D-Aware Scene Video Editing

<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Input Video</td>
        <td>Edited Image</td>
        <td>Generated video</td>
    </tr>

<!-- beach -->
  <tr>
  <td>
    <img src=assets/beach.gif width="200">
  </td>
  <td>
    <img src=examples/beach/editing.png width="200">
  </td>
  <td>
    <img src=assets/beach_edit.gif width="200">
  </td>
  </tr>

<!-- bear -->
<tr>
<td>
    <img src=assets/bear.gif width="200">
</td>
<td>
    <img src=examples/bear/editing.png width="200">
</td>
<td>
    <img src=assets/bear_edit.gif width="200">
</td>
</tr>
<!-- cake -->
<tr>
<td>
    <img src=assets/cake.gif width="200">
</td>
<td>
    <img src=examples/cake/editing.png width="200">
</td>
<td>
    <img src=assets/cake_edit.gif width="200">
</td>
</tr>
<!-- flower -->
<tr>
<td>
    <img src=assets/flower.gif width="200">
</td>
<td>    
    <img src=examples/flower/editing.png width="200">
</td>
<td>
    <img src=assets/flower_edit.gif width="200">
</td>
</tr>
<!-- stone_cat -->
<tr>
<td>
    <img src=assets/stone_cat.gif width="200">
</td>
<td>
    <img src=examples/stone_cat/editing.png width="200">
</td>
<td>
    <img src=assets/stone_cat_edit.gif width="200">
</td>
</tr>

</table>


## 📝 Changelog
- __[2025.08.03]__: 🔥🔥 Release code and model weights.
- __[2025.08.05]__: Launch the project page and update the arXiv preprint.
<br>


## 🧰 Models

|Model|Resolution|GPU Mem. & Inference Time (A100, ddim 50steps)|Checkpoint|
|:---------|:---------|:--------|:--------|
|Sketch3DVE|720x480| ~27G & 53s |[Hugging Face](https://huggingface.co/flashlizard/Sketch3DVE)|

Our method is built based on pretrained [CogVideoX-2b](https://github.com/THUDM/CogVideo) model. We add an additional sketch conditional network for editing. 

Currently, our Sketch3DVE can support generating videos of up to 49 frames with a resolution of 720x480. For editing, we assume the input video has 49 frames with a resolution of 720x480.

The inference time can be reduced by using fewer DDIM steps.


## ⚙️ Setup

### Install Environment via Anaconda (Recommended)
```bash
conda create -n sketch3dve python=3.10
conda activate sketch3dve
pip install -r requirements.txt
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py310_cu118_pyt240.tar.bz2
```
Notably, `diffusers==0.30.1` is required. 

## 💫 Inference
### 1. 3D-Aware Scene Video Editing

Download pretrained Dust3R model [[Download Link](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth)] and DepthAnythingV2 model [[hugging face](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true)] and  LLaVA model [[hugging face](https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b)] and pretrained CogVideoX-2b [[hugging face](https://huggingface.co/THUDM/CogVideoX-2b)] video generation model. Then, modify the `--dust3r_model_path` and `--depthanything_model_path` and `--Llava_model_path` and `--basemodel_ckpt_path` and `--controlnet_ckpt_path`(see the download link above) in examples/xxx/test.sh to corresponding paths. 

Edit example videos. 
```bash
cd examples/beach
sh test.sh
```


## 😉 Citation
Please consider citing our paper if our code is useful:
```bib
@inproceedings{10.1145/3721238.3730623,
author = {Liu, Feng-Lin and Li, Shi-Yang and Cao, Yan-Pei and Fu, Hongbo and Gao, Lin},
title = {Sketch3DVE: Sketch-based 3D-Aware Scene Video Editing},
year = {2025},
isbn = {9798400715402},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
booktitle = {Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
articleno = {152},
numpages = {12},
keywords = {Sketch-based interaction, video generation, video editing, video diffusion models},
location = {
},
series = {SIGGRAPH Conference Papers '25}
}
```


## 🙏 Acknowledgements
We thanks the projects of video generation models [CogVideoX](https://github.com/THUDM/CogVideo) and [ControlNet](https://github.com/lllyasviel/ControlNet) and [Dust3R](https://github.com/naver/dust3r) and [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2). Our code introduction is modified from [ViewCrafter](https://github.com/Drexubery/ViewCrafter) template.

<a name="disc"></a>
## 📢 Disclaimer
Our framework achieves interesting sketch-based 3D-Aware video editing, but due to the variaity of generative video prior, the success rate is not guaranteed. Different random seeds can be tried to generate the best video generation results. 

This project strives to impact the domain of AI-driven video generation positively. Users are granted the freedom to create videos using this tool, but they are expected to comply with local laws and utilize it responsibly. The developers do not assume any responsibility for potential misuse by users.
****