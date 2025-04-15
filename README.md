# Deep Learning for Hazardous Material Detection

This repository provides code for deep learning models that estimate hazardous material release parameters from satellite imagery. The methods were developed as part of the following works:

- 📄 [IEEE SSPD 2023 Paper: Investigation of an End-to-End Neural Architecture for Image-Based Source Term Estimation](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10715304)
- 📄 IEEE SSP 2025 Submission: *A Variational Autoencoder with Semi-Supervised Latent Space for Source Term Estimation*

---

## 🧠 Overview

This repository implements a full pipeline for **Source Term Estimation (STE)** using satellite imagery. It includes both segmentation and inference stages:

- **3D U-Net**  
  Extracts concentration clouds from time-series satellite images (e.g. RGB). Acts as a pre-processing step for STE.

- **Source Term Estimation Models**  
  - **Variational Autoencoder (VAE)**: A novel architecture with dual latent spaces for probabilistic estimation and uncertainty quantification.  
  - **Deterministic Deep Learning (DDL)**: A baseline model for STE without uncertainty estimation.

> Both the VAE and DDL models take the output of the 3D U-Net to estimate key source parameters:
> - Source location (x, y)  
> - Release start time (tₛ)  
> - Wind speed components (uₑ, vₑ)

---

## 📦 Datasets

### 🔬 Simulated Dataset
- Based on a Gaussian puff dispersion model.
- RGB satellite-like images (968×937×3), cropped to 128×128X3.
- Ground truth available for source location, wind speed, and release time.

### 🌍 Real-World Data: Jack Rabbit II Trial 7
- Aerial footage processed with the Segment Anything Model (SAM).
- Cloud shapes extracted via segmentation masks.
- Used for real-world validation of STE models.

### 🚀 Automatic Download
When running the training or evaluation scripts, all required datasets will be **automatically downloaded** to the `Data/` directory. No manual intervention is needed.

---

## 🏗️ Models

### 🔹 3D U-Net
- Takes time-series RGB (or hyperspectral) satellite images.
- Outputs a 3D concentration map for cloud localization over time.

### 🔸 Deterministic Deep Learning (DDL)
- End-to-end CNN for point prediction of source term parameters.
- No uncertainty estimates.

### 🔷 Variational Autoencoder (VAE)
- Dual latent space architecture:
  - **Primary space**: Source location, release time, wind speeds.
  - **Secondary space**: Complementary features (e.g. shape, spread).
- Enables uncertainty quantification and generation of new scenarios.
- Includes LSTM layers to capture temporal evolution.

---

## ⚙️ Getting Started

### Requirements

You can either:

**Option 1: Use Docker (recommended for reproducibility)**
- A ready-to-use `Dockerfile` is provided in the root directory.

**Option 2: Install manually**
- Python 3.8+
- PyTorch ≥ 1.10
- numpy, matplotlib, scipy
- (Optional) CUDA-enabled GPU for training

Use `requirements.txt` to install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ Training

### Train the 3D U-Net model
```bash
python train.py -c configs/unet.yaml --temporal_resolution 20 --save_name 'unet_T=20' --max_epochs 100 --gpu 0

### Train the VAE model
```bash
python train.py -c configs/vae.yaml --temporal_resolution 20 --save_name 'vae_T=20' --max_epochs 100 --gpu 0
```

### Train the DDL model
```bash
python train.py -c configs/ddl.yaml --temporal_resolution 20 --save_name 'ddl_T=20' --max_epochs 100 --gpu 0
```

---

## 🧪 Evaluation

For testing only, pretrained checkpoints used in the papers will be automatically downloaded from the cloud.

Evaluate a trained model on the test set:

```bash
python test.py -c configs/vae.yaml --temporal_resolution 20 --save_name 'vae_T=20'
```

---

## 📝 Citation

If you use this code, please cite the following works:

**[SSP 2025 Submission]**
```
@article{abdulaziz2025vae,
  title={A Variational Autoencoder with Semi-Supervised Latent Space for Source Term Estimation},
  author={Abdullah Abdulaziz, Mike E. Davies, Steve McLaughlin, Yoann Altmann},
  journal={Submitted to IEEE SSP 2025},
  year={2025}
}
```

**[SSPD 2023 Paper]**
```
@inproceedings{abdulaziz2023ddl,
  title={Investigation of an End-to-End Neural Architecture for Image-Based Source Term Estimation},
  author={Abdullah Abdulaziz, Yoann Altmann, Mike E. Davies, Steve McLaughlin},
  booktitle={Sensor Signal Processing for Defence (SSPD)},
  year={2023},
  organization={IEEE}
}
```

---

## 🙌 Acknowledgements

This work was supported by the Engineering and Physical Sciences Research Council (EPSRC) under Grant **EP/T00097X/1**.

---

## 📬 Contact

For questions or collaborations, contact **aazizai@proton.me (Academic: a.abdulaziz@hw.ac.uk)**.
