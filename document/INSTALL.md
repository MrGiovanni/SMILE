# Installation

#### pre-requisites

- Python 3.12 or higher
- PyTorch 2.7 or higher
- CUDA-compatible GPU with CUDA 12.6 or higher
- GPU VRAM > 20 GB is highly recommanded

<details>
<summary style="margin-left: 25px;">Install Anaconda on Linux</summary>
<div style="margin-left: 25px;">
    
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh -b -p ./anaconda3
./anaconda3/bin/conda init
source ~/.bashrc
```
</div>
</details>

<details>
<summary style="margin-left: 25px;">Create A Virtual Environment</summary>
<div style="margin-left: 25px;">
    
```bash
conda create -n smile python=3.12 -y
conda activate smile
```
</div>
</details>

<details>
<summary style="margin-left: 25px;">Merge Updates into Your Local Branch</summary>
<div style="margin-left: 25px;">

```bash
git fetch
git pull
```
</div>
</details>