# DiverseNet: Decision Diversified Semi-supervised Semantic Segmentation Networks for Remote Sensing Imagery

[![arXiv](https://img.shields.io/badge/arXiv-2311.13716-b31b1b.svg)](https://arxiv.org/abs/2311.13716)

This repository contains the official implementation of **DiverseHead**, a lightweight and effective semi-supervised semantic segmentation framework for remote sensing imagery. DiverseHead introduces: Dynamic Freezing and Dropout for perturbation.

The proposed methods achieve competitive performance across four public remote sensing datasets and outperform several state-of-the-art approaches in semi-supervised segmentation.

---

## 📄 Paper

**DiverseNet: Decision Diversified Semi-supervised Semantic Segmentation Networks for Remote Sensing Imagery**  
*Wanli Ma, Oktay Karakus, Paul L. Rosin*  
arXiv:2311.13716  
🔗 [Read the paper](https://arxiv.org/abs/2311.13716)

---

## 🧰 Features

- ✅ Lightweight model design for low-memory environments
- ✅ Enhanced pseudo-label generation with model and decision diversity
- ✅ Modular structure, easy to integrate into existing pipelines
- ✅ Comprehensive evaluation on multiple benchmark datasets

---

## 📦 Datasets

The following remote sensing datasets were used:

- ISPRS Potsdam
- DFC2020
- RoadNet
- Massachusetts Buildings

---

## 🚀 Installation

### Prerequisites

- Python >= 3.7
- PyTorch >= 1.8
- CUDA >= 10.2 (for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/WANLIMA-CARDIFF/DiverseNet.git
cd DiverseNet

# (Optional) Create a conda environment
conda create -n diversenet python=3.8 -y
conda activate diversenet

# Install dependencies
pip install -r requirements.txt

```

## 🏃‍♂️ Usage

## 📦 Pretrained Models

We provide the following pretrained models:

| Model          | Description            | Download Link |
|----------------|------------------------|----------------|
| DiverseHead DF | Trained on RoadNet   | [Download](https://drive.google.com/file/d/1mScrNmveUWpM8gALMCGOL2H1KcI08ATS/view?usp=sharing) |
| DiverseHead DT | Trained on RoadNet   | [Download](https://drive.google.com/file/d/1caDKR2YjAnRstBQipCEh4Gbd-QOazZ3w/view?usp=sharing) |

After downloading, place the `.pth` files into the root directory.

### Test with DiverseHead

```bash
python test.py 
```
### Train with DiverseHead

```bash
python train.py 
```
## 📬 Contact

For questions or collaboration opportunities:

- **Wanli Ma** – [maw13@cardiff.ac.uk](mailto:maw13@cardiff.ac.uk)  

