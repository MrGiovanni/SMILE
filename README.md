<div align="center">
  <img src="document/pdac_example.png" alt="SMILE" width=100%>
</div>

<h1 align="center" style="font-size: 60px; margin-bottom: 4px">Anatomy-Aware Contrast Enhancement</h1>

<div align="center">


[![smile dataset](https://img.shields.io/badge/SMILE-Dataset-FF4040.svg)](https://github.com/MrGiovanni/SMILE?tab=readme-ov-file#dataset)
[![smile benchmark](https://img.shields.io/badge/SMILE-Benchmark-FF4040.svg)](https://github.com/MrGiovanni/SMILE?tab=readme-ov-file#benchmark)
[![HuggingFace](https://img.shields.io/badge/SMILE-Model-FF4040.svg)](https://github.com/MrGiovanni/SMILE?tab=readme-ov-file#model) <br/>
![visitors](https://visitor-badge.laobi.icu/badge?page_id=MrGiovanni/SMILE&left_color=%2363C7E6&right_color=%23CEE75F)
[![GitHub Repo stars](https://img.shields.io/github/stars/MrGiovanni/SMILE?style=social)](https://github.com/MrGiovanni/SMILE/stargazers) 
<a href="https://twitter.com/bodymaps317">
        <img src="https://img.shields.io/twitter/follow/BodyMaps?style=social" alt="Follow on Twitter" />
</a><br/>  

</div>


We present **SMILE** (Super Modality Image Learning and Enhancement), an anatomy-aware diffusion model for clinically reliable CT contrast enhancement. SMILE achieves significant improvements: **+14.2% SSIM**, **+20.6% PSNR**, **+50% FID**, and enables cancer detection from non-contrast CT scans with **+10% F1 score** improvement.

<a id="smile-paper"></a>
# Paper
<b>See More, Change Less: Anatomy-Aware Diffusion for Contrast Enhancement</b> <br/>
[Junqi Liu](https://scholar.google.com/citations?hl=en&authuser=1&user=4Xpspl0AAAAJ), [Zejun Wu](https://scholar.google.com/citations?hl=en&user=s2umIj8AAAAJ), [Pedro R. A. S. Bassi](https://scholar.google.com/citations?hl=en&user=NftgL6gAAAAJ), [Xinze Zhou](), [Wenxuan Li](https://scholar.google.com/citations?hl=en&user=tpNZM2YAAAAJ), [Ibrahim E. Hamamci](https://scholar.google.com/citations?hl=en&user=7bN36N0AAAAJ), [Sezgin Er](https://scholar.google.com/citations?hl=en&user=_vr-hRkAAAAJ), [Tianyu Lin](https://scholar.google.com/citations?hl=en&user=eHJYs-IAAAAJ), [Yi Luo](), [Szymon Płotka](https://scholar.google.com/citations?hl=en&authuser=1&user=g9sWRN0AAAAJ), [Bjoern Menze](https://scholar.google.com/citations?hl=en&authuser=1&user=Kv2QrQgAAAAJ), [Daguang Xu](https://scholar.google.com/citations?hl=en&authuser=1&user=r_VHYHAAAAAJ), [Kai Ding](https://scholar.google.com/citations?hl=en&authuser=1&user=OvpsAYgAAAAJ), [Kang Wang](https://radiology.ucsf.edu/people/kang-wang), [Yang Yang](https://scholar.google.com/citations?hl=en&authuser=1&user=6XsJUBIAAAAJ), [Yucheng Tang](https://scholar.google.com/citations?hl=en&authuser=1&user=0xheliUAAAAJ), [Alan Yuille](https://www.cs.jhu.edu/~ayuille/), [Zongwei Zhou](https://www.zongweiz.com/)<sup>★</sup> <br/>
Johns Hopkins University <br/>
<a href='https://www.zongweiz.com/dataset'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://www.cs.jhu.edu/~zongwei/preprint/liu2025see.pdf'><img src='https://img.shields.io/badge/Paper-PDF-purple'></a>

# Model

#### 1 | Install

To set up environment, see [INSTALL.md](https://github.com/MrGiovanni/SMILE/blob/main/document/INSTALL.md) for details.

```bash
git clone https://github.com/MrGiovanni/SMILE.git
cd SMILE
while read requirement; do
    pip install "$requirement" || echo "Failed to install $requirement, skipping..."
done < requirements.txt
```
#### 2 | Download checkpoint

```bash
bash download_ckpts.sh
```

#### 3 | Direct inference

###### 3.1 ｜ We provide demo data for quick testing

```bash
bash download_demo_dataset.sh
bash inference.sh
```
The enhancement results are in the `./out` folder.

###### 3.2 ｜ Test on your own data

First, save your data folder (e.g., `PanTS`) in `./data` using the same format as our demo data. 

Second, modify the parameters in the `inference.sh`:

1. `Dataset_Name`: name of your own dataset name (e.g., `PanTS`).

    ```bash
      data
      └──PanTS/
          ├── PanTS_000001/
          │     └── ct.nii.gz
          ├── PanTS_000002/
          │     └── ct.nii.gz
    ```
2. `TARGETS` (optional): enhancement targets. Default as `("arterial" "venous" "delayed")`.
3. `GUIDE_CSV` (optional): .csv file to guide model to inference on specific cases. An example:<br>
```bash
Inference ID
BDMAP_xxxx01
BDMAP_xxxx02
BDMAP_xxxx03
BDMAP_xxxx04
...
```

# Benchmark

#### 1 | Image enhancement methods

> [!NOTE]
> We are calling for more baseline methods. 

| model  | paper | github | SSIM | PSNR | FID | Intensity Correlation |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| Pix2Pix | [![arXiv](https://img.shields.io/badge/arXiv-1611.07004-FF4040.svg)](https://arxiv.org/abs/1611.07004) | [![GitHub stars](https://img.shields.io/github/stars/phillipi/pix2pix.svg?logo=github&label=Stars)](https://github.com/phillipi/pix2pix) | 60.7 | 18.8 | 299.7 | 0.26
| CycleGAN | [![arXiv](https://img.shields.io/badge/arXiv-1703.10593-FF4040.svg)](https://arxiv.org/abs/1703.10593) | [![GitHub stars](https://img.shields.io/github/stars/junyanz/CycleGAN.svg?logo=github&label=Stars)](https://github.com/junyanz/CycleGAN) | 71.9 | 18.2 | 271.1 | 0.09
| DALL-E | [![arXiv](https://img.shields.io/badge/arXiv-2012.0984-FF4040.svg)](https://arxiv.org/abs/2012.09841) | [![GitHub stars](https://img.shields.io/github/stars/hojonathanho/diffusion.svg?logo=github&label=Stars)](https://github.com/hojonathanho/diffusion) | 51.4 | 16.3 | 423.7 | 0.71
| Stable Diffusion | [![arXiv](https://img.shields.io/badge/arXiv-2112.10752-FF4040.svg)](https://arxiv.org/abs/2112.10752) | [![GitHub stars](https://img.shields.io/github/stars/CompVis/stable-diffusion.svg?logo=github&label=Stars)](https://github.com/CompVis/stable-diffusion) | 64.6 | 16.0 | 406.3 | 0.45
| CUT | [![arXiv](https://img.shields.io/badge/arXiv-2007.15651-FF4040.svg)](https://arxiv.org/abs/2007.15651) | [![GitHub stars](https://img.shields.io/github/stars/lllyasviel/ControlNet.svg?logo=github&label=Stars)](https://github.com/lllyasviel/ControlNet) | 75.4 | 21.4 | 269.5 | 0.06
| SMILE | [![arXiv](https://img.shields.io/badge/arXiv-2512.07251-FF4040.svg)](https://www.arxiv.org/abs/2512.07251) | [![GitHub stars](https://img.shields.io/github/stars/MrGiovanni/SMILE.svg?logo=github&label=Stars)](https://github.com/MrGiovanni/SMILE) | 86.1 | 25.8 | 133.4 |0.95

#### 2 | Commerical AI models

<div align="center">
  <img src="document/Twitter.png" alt="SMILE" width=100%>
</div>



# Dataset

Our work further includes **CTVerse**, a large-scale multi-phase CT dataset containing **477** patients from **112** hospitals, with all four contrast phases (non-contrast, arterial, venous, and delay). 

```shell
# This work is currently under peer review, but early access is available!
# To request the CTVerse dataset files, please email Zongwei Zhou at zzhou82@jh.edu
```

<!-- #### Official training set
- CTVerse-tr (*n*=382)

#### Official *in-distribution* test set 

- CTVerse-te (*n*=95)

#### Four contrast phases per patient

> [!NOTE]
> Each patient has four contrast phases captured at different timepoints:
> - **Non-contrast (N)**: Baseline scan before contrast injection (0s)
> - **Arterial (A)**: Highlights arteries and early vascular structures (30s)
> - **Venous (V)**: Enhances organs such as liver and spleen (75s)
> - **Delay (D)**: Shows mainly urinary system (>180s)

#### Anatomical annotations

- **88 anatomical structures** including organs, vessels, bones, and disease regions
- **159,632 three-dimensional masks** total across all patients and phases
- Annotations for **pancreatic, liver, and kidney tumors** -->

# Citation

```
@article{liu2025see,
  title={See More, Change Less: Anatomy-Aware Diffusion for Contrast Enhancement},
  author={Liu, Junqi and Wu, Zejun and Bassi, Pedro RAS and Zhou, Xinze and Li, Wenxuan and Hamamci, Ibrahim E and Er, Sezgin and Lin, Tianyu and Luo, Yi and Płotka, Szymon and others},
  journal={arXiv preprint arXiv:https://www.arxiv.org/abs/2512.07251},
  year={2025},
  url={https://github.com/MrGiovanni/SMILE}
}
```

# Acknowledgement

This work was supported by the Lustgarten Foundation for Pancreatic Cancer Research, the Patrick J. McGovern Foundation Award, and the National Institutes of Health (NIH) under Award Number R01EB037669. We would like to thank the Johns Hopkins Research IT team in [IT@JH](https://researchit.jhu.edu/) for their support and infrastructure resources where some of these analyses were conducted; especially [DISCOVERY HPC](https://researchit.jhu.edu/research-hpc/). Paper content is covered by patents pending.
