<div align="center">
  <img src="document/fig_teaser.jpg" alt="SMILE" width=100%>
</div>

<h1 align="center" style="font-size: 60px; margin-bottom: 4px">😊 SMILE: Anatomy-Aware Contrast Enhancement</h1>

<div align="center">


[![smile dataset](https://img.shields.io/badge/SMILE-Dataset-FF4040.svg)](https://github.com/MrGiovanni/SMILE?tab=readme-ov-file#ctverse-dataset)
[![smile benchmark](https://img.shields.io/badge/SMILE-Benchmark-FF4040.svg)](https://github.com/MrGiovanni/SMILE?tab=readme-ov-file#smile-benchmark)
[![smile model](https://img.shields.io/badge/SMILE-Model-FF4040.svg)](https://github.com/MrGiovanni/SMILE?tab=readme-ov-file#smile-model) <br/>
![visitors](https://visitor-badge.laobi.icu/badge?page_id=MrGiovanni/SMILE&left_color=%2363C7E6&right_color=%23CEE75F)
[![GitHub Repo stars](https://img.shields.io/github/stars/MrGiovanni/SMILE?style=social)](https://github.com/MrGiovanni/SMILE/stargazers) 
<a href="https://twitter.com/bodymaps317">
        <img src="https://img.shields.io/twitter/follow/BodyMaps?style=social" alt="Follow on Twitter" />
</a><br/>  

</div>


We present **SMILE** (Super Modality Image Learning and Enhancement), an anatomy-aware diffusion model for clinically reliable CT contrast enhancement. Unlike existing generative models that often over-edit and distort anatomical structures, SMILE learns the spatial and physiological relationships between organs and their contrast uptake, enhancing only clinically relevant regions while keeping others unchanged.

Our work includes **CTVerse**, a large-scale multi-phase CT dataset containing **477** patients from **112** hospitals, with all four contrast phases (non-contrast, arterial, venous, and delay). Each scan is annotated with **88** anatomical structures and tumors, resulting in **159,632** three-dimensional masks.

SMILE achieves significant improvements: **+14.2% SSIM**, **+20.6% PSNR**, **+50% FID**, and enables cancer detection from non-contrast CT scans with **+10% F1 score** improvement.

# 📰 News & Updates
Major updates and announcements are shown below. Scroll for full timeline.

🔥 [2025-11] **Repository Launch** -- SMILE v0.1 is now live !!! We are building the comprehensive diffusion framework, that can enhance CT images precisely and **clinically meaningful**.

🔥 [2025-11] **New Version Updated** -- SMILE v0.2 is now available !! Compared to the initial version, v0.2 improves greatly in removing the small artifacts and organ HU range. See [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow.svg?logo=huggingface)](https://huggingface.co/your-model-link) for model config details.

🤖 [2025-12] **Better Segmenter** -- We provide a better segmenter (+5 avg. DSC), that trained with more dynamic data! The model is trained on public PanTS dataset [![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/MrGiovanni/PanTS), and the model is now online: [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow.svg?logo=huggingface)](https://huggingface.co/your-model-link).

🚀 [Ongoing] **New Version Preparing** -- SMILE v0.3 is undergoing fine-tuning process and will be made available soon! This version expects to make organ substructures such as kidney cortex more realistic 📈.

# Overview
* 🎯 [<u>**Paper**</u>](#smile-paper)
* 💯 [<u>**SMILE Benchmarks**</u>](#smile-benchmarks)
* 🦾 [<u>**SMILE Guidebooks**</u>](#smile-guidebook)
* 🔬 [<u>**Surveys of Generative Models in Medial Imageing**</u>](#smile-survey)
* 🌍 [<u>**CTVerse Dataset**</u>](#CTVerse-dataset)
* 👩‍🏫 [<u>**Citations**</u>](#smile-citations)


<a id="smile-paper"></a>
# Paper
<b>See More, Change Less: Anatomy-Aware Diffusion for Contrast Enhancement</b> <br/>
[Junqi Liu](https://scholar.google.com/citations?hl=en&authuser=1&user=4Xpspl0AAAAJ), [Zejun Wu](), [Pedro R. A. S. Bassi](), [Xinze Zhou](), [Wenxuan Li](), [Ibrahim E. Hamamci](), [Sezgin Er](), [Tianyu Lin](), [Yi Luo](), [Szymon Płotka](https://scholar.google.com/citations?hl=en&authuser=1&user=g9sWRN0AAAAJ), [Bjoern Menze](https://scholar.google.com/citations?hl=en&authuser=1&user=Kv2QrQgAAAAJ), [Daguang Xu](https://scholar.google.com/citations?hl=en&authuser=1&user=r_VHYHAAAAAJ), [Kai Ding](https://scholar.google.com/citations?hl=en&authuser=1&user=OvpsAYgAAAAJ), [Kang Wang](https://radiology.ucsf.edu/people/kang-wang), [Yang Yang](https://scholar.google.com/citations?hl=en&authuser=1&user=6XsJUBIAAAAJ), [Yucheng Tang](https://scholar.google.com/citations?hl=en&authuser=1&user=0xheliUAAAAJ), [Alan Yuille](https://www.cs.jhu.edu/~ayuille/), [Zongwei Zhou](https://www.zongweiz.com/)<sup>★</sup> <br/>
Johns Hopkins University, University of Copenhagen, University of Virginia, University of Bologna, and others

<a href='https://www.zongweiz.com/dataset'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://www.cs.jhu.edu/~zongwei/preprint/liu2025see.pdf'><img src='https://img.shields.io/badge/Paper-PDF-purple'></a>

<a id="smile-benchmarks"></a>

# SMILE Benchmark (official in-distribution test set)

> [!NOTE]
> We are calling for more baseline methods. 

| model  | paper | github | SSIM | PSNR | FID | Intensity Correlation |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| Pix2Pix | [![arXiv](https://img.shields.io/badge/arXiv-1611.07004-FF4040.svg)](https://arxiv.org/abs/1611.07004) | [![GitHub stars](https://img.shields.io/github/stars/phillipi/pix2pix.svg?logo=github&label=Stars)](https://github.com/phillipi/pix2pix) | 60.7 | 18.8 | 299.7 | 0.26
| CycleGAN | [![arXiv](https://img.shields.io/badge/arXiv-1703.10593-FF4040.svg)](https://arxiv.org/abs/1703.10593) | [![GitHub stars](https://img.shields.io/github/stars/junyanz/CycleGAN.svg?logo=github&label=Stars)](https://github.com/junyanz/CycleGAN) | 71.9 | 18.2 | 271.1 | 0.09
| DDPM | [![arXiv](https://img.shields.io/badge/arXiv-2006.11239-FF4040.svg)](https://arxiv.org/abs/2006.11239) | [![GitHub stars](https://img.shields.io/github/stars/hojonathanho/diffusion.svg?logo=github&label=Stars)](https://github.com/hojonathanho/diffusion)
| Stable Diffusion | [![arXiv](https://img.shields.io/badge/arXiv-2112.10752-FF4040.svg)](https://arxiv.org/abs/2112.10752) | [![GitHub stars](https://img.shields.io/github/stars/CompVis/stable-diffusion.svg?logo=github&label=Stars)](https://github.com/CompVis/stable-diffusion) | 64.6 | 16.0 | 406.3 | 0.45
| ControlNet | [![arXiv](https://img.shields.io/badge/arXiv-2302.05543-FF4040.svg)](https://arxiv.org/abs/2302.05543) | [![GitHub stars](https://img.shields.io/github/stars/lllyasviel/ControlNet.svg?logo=github&label=Stars)](https://github.com/lllyasviel/ControlNet)
| SMILE | [![arXiv](https://img.shields.io/badge/arXiv-TBD-FF4040.svg)](https://arxiv.org/abs/TBD) | [![GitHub stars](https://img.shields.io/github/stars/MrGiovanni/SMILE.svg?logo=github&label=Stars)](https://github.com/MrGiovanni/SMILE) | 86.1 | 25.8 | 133.4 |0.95

## Comparison to current commerical AI models
<div align="center">
  <img src="document/Twitter.png" alt="SMILE" width=100%>
</div>


<a id="smile-guidebook"></a>
# SMILE Model 

> [!NOTE]
> Model checkpoints will be released upon paper acceptance. Stay tuned!

**Key Features:**
- **Structure-aware supervision** that guides enhancements to follow realistic organ boundaries and contrast dynamics
- **Registration-free generation** that learns directly from unaligned multi-phase CT scans without voxel-wise registration
- **Unified inference** for efficient, consistent enhancement across multiple contrast phases in a single diffusion pipeline

**Clinical Applications:**
- Opportunistic cancer screening from non-contrast CT scans
- Enhanced tumor detection without additional contrast injection
- Accessible diagnostic imaging for patients who cannot receive contrast agents

## Installation

### Prerequisites

- Python 3.12 or higher
- PyTorch 2.7 or higher
- CUDA-compatible GPU with CUDA 12.6 or higher
- GPU VRAM > 20 GB is highly recommanded
- A100, H100 or higher are strongly suggested 😈

### **Create a new Conda environment:**

```bash
conda create -n smile python=3.12
conda deactivate
conda activate smile
```

### **Clone the repository and install the package:**

```bash
git clone https://github.com/MrGiovanni/SMILE.git
cd SMILE
pip install -r requirements.txt
```

## Getting Started
**(1) multiple-CT infernce / dataset inference:**
```bash
bash inference_multiple_CT.sh
```
[See detailed instructions ↓](#multi-ct-inference)


**(2) single-CT case infernce:**
```bash
bash inference.sh --gpu_id 0 --source non-contrast --target arterial,venous,delayed --patient_id RS-GIST-121
```
[See detailed instructions ↓](#multi-ct-inference)


## SMILE Guide Book
### step 0. download SMILE models checkpoints
To inference SMILE, please first download the pre-trained checkpoints for VAE, Classifier and Segmenter, as well as the trained SMILE model.

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow.svg?logo=huggingface)](https://huggingface.co/your-model-link)

<details> 
<summary>download model checkpoints:</summary>

## Download the Super-VAE checkpoints
```bash
hf download CVPR-SMILE/SMILE_mini --include="autoencoder/*" --local-dir "./ckpt" 
```

## Download the Classification model:
```bash
hf download CVPR-SMILE/SMILE_mini --include="classifier/*" --local-dir="./ckpt"
```

## Download the Segmentation model:
```bash
hf download CVPR-SMILE/SMILE_mini --include="segmenter/*" --local-dir="./ckpt"
```

## Download the SMILE checkpoints
```bash
hf download CVPR-SMILE/SMILE_mini --include="SMILE_v0.2/*" --local-dir="./ckpt"
```

</details>

### step 1. prepare the data:

The data for inference is expected to be organized into BDMAP form, as below:

```bash
  DATA_FOLDER
    DATASET_NAME/
    ├── BDMAP_00012345/
    │     └── ct.nii.gz
    ├── BDMAP_00067890/
    │     └── ct.nii.gz
```

<a id="multi-ct-inference"></a>
### step 2. choose inference mode:
1. **multiple-CT infernce / dataset inference:**
    ```bash
    cd SMILEModel
    bash inference_multiple_CT.sh
    ```
    <details> <summary>Click to view Configuration Details</summary>

    1. Edit the User Defined section in the script to point to your data, model, and guide CSV:
          ```bash
          # ================ User Defined ================ #
          MODEL_CKPT_NAME="SMILE/Version (default as v0.2)"
          DATA_FOLDER="/path/to/nifti/dataset/parent/folder"
          DATASET_NAME="/dataset/name/as/nnunetv2/from"
          # Enhancement targets
          TARGETS=("arterial" "venous" "delayed")
          GUIDE_CSV="/nifti/you/want/to/inference"
          # ============================================== #
          ```

        `MODEL_CKPT_NAME`: Version of SMILE model, deault as v0.2 (stable and fast).

        `DARA_FOLDER`: Parental folder to the inference dataset.

        `DATASET_NAME`: Name of the inference dataset, desired in *nnunetv2* form.

        `TARGETS`: One or multiple enhancement targets (comma-separated).  
          Options: `non-contrast`, `arterial`, `venous`, `delayed`.

        `GUIDE_CSV`: Case list CSV, containing the cases to inference.


    2. The outputs are supposed to be stored in:
        ```bash
        SMILE/out/DATASETNAME-SMILE_Version
        # example: ../out/Dataset101_Example-SMILE_v0.2
        ```

    </details>


2. **single-CT case infernce:**
    ```bash
    cd SMILEModel
    bash inference.sh \
        --gpu_id 0 \
        --source non-contrast \
        --target arterial,venous,delayed \
        --patient_id Example-1
    ```

    <details> <summary>Click to view Configuration Details</summary>

      1. Edit the argument section when running the script to specify the GPU, source phase, target phases, patient ID, and model:

          ```bash
              --gpu_id 0 \
              --source non-contrast \
              --target arterial,venous,delayed \
              --patient_id Example-1
          ```
      - **`--gpu_id`**: GPU index used for inference (default = `0` if omitted).

      - **`--source`**: Input CT phase.  
          Options: `non-contrast`, `arterial`, `venous`, `delayed`.

      - **`--target`**: One or multiple enhancement targets (comma-separated).  
          Options: `non-contrast`, `arterial`, `venous`, `delayed`.

      - **`--patient_id`**: Patient identifier following BDMAP naming (e.g., `RS-GIST-121`).

      - **`--model`**: SMILE model version to load.  
          Default: `SMILE_v0.2.2-a100-200k`.

      2. The script automatically finds the correct input NIfTI file, loads the fine-tuned UNet + VAE, and performs multi-phase translation in a single call.

      3. The outputs are supposed to be stored in:
          ```bash
          SMILE/out/DATASETNAME-SMILE_Version
          # example: ../out/Dataset101_Example-SMILE_v0.2
          ```
    </details>

<a id="smile-survey"></a>

# Surveys 

## 💉 Surveys of Generative Models in Medical Imaging

[👩‍⚕️] models specificly designed for medical imaging use.

[⭐️] surveys for generative models in medical imaging.


### GAN-based Generative Models
- **Pix2Pix**, “Image-to-Image Translation with Conditional Adversarial Networks”.  [![arXiv](https://img.shields.io/badge/arXiv-1611.07004-b31b1b.svg)](https://arxiv.org/abs/1611.07004) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://phillipi.github.io/pix2pix/) 

- **CycleGAN**, “CycleGAN for Unpaired Medical Image Translation”.  [![arXiv](https://img.shields.io/badge/arXiv-1703.10593-b31b1b.svg)](https://arxiv.org/abs/1703.10593) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://junyanz.github.io/CycleGAN/) 


- [👩‍⚕️] **CyTran**, “CyTran: Cycle-Consistent Transformers for Medical Image Translation”.   [![arXiv](https://img.shields.io/badge/arXiv-2301.12345-b31b1b.svg)](https://arxiv.org/abs/2110.06400)  

- **CUT**, “Contrastive Unpaired Translation for Medical Applications”.  [![arXiv](https://img.shields.io/badge/arXiv-2007.15651-b31b1b.svg)](https://arxiv.org/abs/2007.15651) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://taesung.me/ContrastiveUnpairedTranslation/) 


### VAE-based Generative Models
- **DALL-E / VQ-VAE**, “Taming Transformers for Diverse Image Generation”.  [![arXiv](https://img.shields.io/badge/arXiv-2012.09841-b31b1b.svg)](https://arxiv.org/abs/2012.09841) [![Website](https://img.shields.io/badge/Website-Link-blue)](https://openai.com/index/dall-e/) 


### Diffusion-based Generative Models
- [👩‍⚕️] **MedDiffusion**, “Medical Diffusion: Denoising Diffusion Probabilistic Models for 3D Medical Image Generation”.  [![arXiv](https://img.shields.io/badge/arXiv-2307.12345-b31b1b.svg)](https://arxiv.org/abs/2211.03364)


### Comprehensive Surveys
> [!NOTE]
> We are calling for more outstanding surveys 😊. 
- [⭐️] **Diffusion Models for Medical Image Analysis: A Comprehensive Survey**.  
  [![arXiv](https://img.shields.io/badge/arXiv-2207.10454-b31b1b.svg)](https://arxiv.org/abs/2211.07804)

- [⭐️] **Generative ai for medical imaging: extending the monai framework**.
  [![arXiv](https://img.shields.io/badge/arXiv-2303.09334-b31b1b.svg)](https://arxiv.org/abs/2307.15208)





<a id="CTVerse-dataset"></a>

# CTVerse Dataset

```shell
git clone https://github.com/MrGiovanni/SMILE.git
cd SMILE
bash download_CTVerse_data.sh # It needs storage for multi-phase CT scans
bash download_CTVerse_label.sh 
# This work is currently under peer review, but early access is available!
# To request the CTVerse dataset files, please email Zongwei Zhou at zzhou82@jh.edu
```

#### Official training set
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
- Annotations for **pancreatic, liver, and kidney tumors**






<a id="smile-citations"></a>

# Citation

```
@article{liu2025see,
  title={See More, Change Less: Anatomy-Aware Diffusion for Contrast Enhancement},
  author={Liu, Junqi and Wu, Zejun and Bassi, Pedro RAS and Zhou, Xinze and Li, Wenxuan and Hamamci, Ibrahim E and Er, Sezgin and Lin, Tianyu and Luo, Yi and Płotka, Szymon and others},
  journal={arXiv preprint arXiv:TBD},
  year={2025},
  url={https://github.com/MrGiovanni/SMILE}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MrGiovanni/SMILE?tab=readme-ov-file&type=date&legend=top-left)](https://www.star-history.com/#MrGiovanni/SMILE?tab=readme-ov-file&type=date&legend=top-left)


# Acknowledgement

This work was supported by the Lustgarten Foundation for Pancreatic Cancer Research, the Patrick J. McGovern Foundation Award, and the National Institutes of Health (NIH) under Award Number R01EB037669. We would like to thank the Johns Hopkins Research IT team in [IT@JH](https://researchit.jhu.edu/) for their support and infrastructure resources where some of these analyses were conducted; especially [DISCOVERY HPC](https://researchit.jhu.edu/research-hpc/). Paper content is covered by patents pending.