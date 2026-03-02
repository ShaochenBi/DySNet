# [DySNet: Dynamic Stream Network](https://github.com/ShaochenBi/DySNet)

[![header](https://capsule-render.vercel.app/api?type=rect&height=120&color=gradient&text=DySNet%20-%20Dynamic%20Stream%20Network&fontAlign=50&reversal=true&textBg=false&fontAlignY=37&fontSize=40&desc=For%20Combinatorial%20Explosion%20In%20Medical%20Registration&descSize=35&descAlign=50&descAlignY=75)](https://github.com/ShaochenBi/DySNet)

---
[![Paper](https://img.shields.io/badge/CVPR-2026-blue.svg)](https://arxiv.org/abs/2506.20850) 
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Dynamic Stream Network (DySNet)** is a novel dynamic modeling framework designed to tackle the **Combinatorial Explosion** challenge in Deformable Medical Image Registration (DMIR). By introducing dynamic receptive fields and weights, DySNet effectively eliminates interfering features and captures potential feature relationships.

---

> [**Dynamic Stream Network for Combinatorial Explosion Problem in Deformable Medical Image Registration**](https://github.com/ShaochenBi/DySNet)  
> [Shaochen Bi](mailto:bisc0507@163.com), [Weiming Wang](mailto:wmwang@hkmu.edu.hk), [Yuting He](mailto:yuting.he4@case.edu), [Hao Chen](mailto:jhc@ust.hk)  
> **Accepted to: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026**

---

## ✨ Highlights

> **DySNet** — A modular dynamic modeling network that adaptively adjusts search spaces and directions through **Adaptive Stream Basin (AdSB)** and **Dynamic Stream Attention (DySA)** modules.

---

### 🌪️ 1. Tackling Combinatorial Explosion
In dual-input tasks like registration, the number of feature combinations grows exponentially with resolution. Static receptive fields often introduce irrelevant features. **DySNet** transforms static modeling into a "stream-like" dynamic process, significantly narrowing the search space.

<div align="center">
  <img src="fig/concept.png" width="80%" alt="Combinatorial Explosion Solution"/>
  <br>
  <em>From static to dynamic: DySNet narrows the search space and accurately locates feature correspondences.</em>
</div>

---

### 🌊 2. Adaptive Stream Basin (AdSB)
Inspired by how streams flow through a basin, **AdSB** predicts offsets to adjust the shape of the receptive field dynamically.
* **Deformable Receptive Fields**: Adaptively reshapes the sampling window based on anatomical structure differences.
* **Interference Elimination**: Filters out irrelevant feature combinations within the search space.

---

### 🎯 3. Dynamic Stream Attention (DySA)
Built upon the dynamic fields provided by AdSB, **DySA** introduces point-to-point attention mechanisms.
* **Dynamic Weights**: Calculates spatial weights in real-time based on feature similarity rather than fixed parameters.
* **Precise Alignment**: Adjusts search directions to capture the most accurate correspondences in the deformed space.

<div align="center">
  <img src="fig/framework.png" width="100%" alt="DySNet Framework"/>
  <br>
  <em>The DySNet Architecture: A symmetrical registration network built with Dynamic Stream Blocks (DSB).</em>
</div>

---

### 🧬 4. Versatility & Generalization
* **Plug-and-Play**: Modular design that integrates seamlessly into frameworks like Xmorpher and ModeT.
* **Multi-Dimensional**: Supports both **2D** and **3D** registration (CT Bone, MRI Brain, Cardiac CT).
* **Robust Performance**: Achieves SOTA performance with significant gains in DSC across multiple benchmarks.

---

### 📈 5. Quantitative Benchmarks
DySNet demonstrates superior performance across three major tasks:
| Task | Dataset | Metric (DSC) | Improvement |
|:---:|:---:|:---:|:---:|
| **3D Cardiac** | CT | **84.1%** | +7.2% vs VoxelMorph |
| **3D Brain** | MRI | **79.7%** | SOTA Performance |
| **2D Brain** | MRI | **83.0%** | Robust Generalization |

---

## 🛣️ Roadmap

| Status | Feature / Goal | Description |
|:------:|:----------------|:-------------|
| ✅ | **Core Architecture** | Development of the DSB-based dynamic feature modeling network |
| ✅ | **AdSB & DySA Modules** | Implementation of dynamic receptive fields and adaptive weights |
| ✅ | **Multi-Framework Integration** | Instantiation of DySNet-X (Xmorpher) and DySNet-M (ModeT) |
| 🔜 | **Source Code Release** | Official PyTorch implementation coming soon |
| 🔜 | **Pre-trained Weights** | Checkpoints for 3D Brain/Cardiac datasets |
| 🚧 | **Cross-Modal Support** | Extension to PET/CT and MRI/US registration tasks |
| 💡 | **Memory Optimization** | Reducing GPU memory footprint for ultra-high-res 3D volumes |

---

## 💖 Acknowledgements
We thank the computational resources provided by **HKUST**, **HKMU**, and **Case Western Reserve University**. We also thank the open-source community for providing the foundational frameworks that made DySNet possible. Special thanks to everyone who stars ⭐ this repository!

---

## 💡 Citation
If you find our work useful for your research, please cite our paper:

```bibtex
@InProceedings{Bi_2026_CVPR,
    author    = {Bi, Shaochen and Wang, Weiming and He, Yuting and Chen, Hao},
    title     = {Dynamic Stream Network for Combinatorial Explosion Problem in Deformable Medical Image Registration},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026}
}
