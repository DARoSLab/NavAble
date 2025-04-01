# NavAble
Synthetic data generation for accessibility objects/landmarks using UE4

# 🦯 NavAble: A Synthetic Dataset for Accessible Object Perception in Guide Robot Applications

[![Paper](https://img.shields.io/badge/Paper-Coming_Soon-blue)](#)
[![License](https://img.shields.io/github/license/YOUR_ORG/navable)](LICENSE)
[![GitHub contributors](https://img.shields.io/github/contributors/YOUR_ORG/navable)](https://github.com/YOUR_ORG/navable/graphs/contributors)

<div align="center">
<strong>
[
<a href="#dataset-overview">Dataset Overview</a> |
<a href="#data-generation">Data Generation</a> |
<a href="#download">Download</a> |
<a href="#usage">Usage</a> |
<a href="#citation">Citation</a>
]
</strong>
</div>

---

**NavAble** is a high-quality **synthetic dataset** built in **Unreal Engine 4 (UE4)** to support **visual perception of accessibility-related objects** (e.g., tactile paving, accessible pedestrian signals, guide rails, curbs) for **guide robots assisting blind and low-vision people**.

It is designed to advance research in:
- Accessible urban navigation  
- Object detection and segmentation  
- Sim-to-real generalization  
- Foundation models for sidewalk understanding

<p align="center">
  <img src="documentation/source/figs/navable_teaser.gif" width="100%">
</p>

---

## 🔍 Dataset Overview

Each sample in **NavAble** includes:

| Modality        | Description                                |
|-----------------|--------------------------------------------|
| RGB Images      | Synthetic urban scenes with realistic lighting |
| Segmentation    | Pixel-level semantic labels for accessible objects |
| Depth Maps      | Aligned depth images from simulated sensors |
| Metadata        | JSON files containing bounding boxes, scene parameters, and object annotations |
| Camera Pose     | Intrinsics and extrinsics in standard format |

**Key Accessible Object Classes:**
- Tactile paving  
- Pedestrian crossing buttons  
- Walk/don’t walk signals  
- Accessible ramps  
- Guide rails  
- Crosswalk lines  

---

## 🛠 Data Generation

All data is generated using our custom pipeline built on **Unreal Engine 4**, which provides:

- Procedurally generated city blocks with accessibility features  
- Diverse lighting and weather conditions  
- Camera motion simulating robot navigation paths  
- Simulated stereo + depth cameras at pedestrian height  

For data generation details, please refer to [`/scripts/generation_pipeline/`](scripts/generation_pipeline/) and see the documentation in [`/docs`](docs/).

---

## 📦 Download

You can request access to the full dataset by filling out the following form:

📄 **[Request Access Form](https://forms.office.com/...)**  
🔑 Access credentials will be emailed upon approval.

Alternatively, we provide a small sample subset:

```bash
wget https://YOUR_SERVER/navable_sample_subset.zip
unzip navable_sample_subset.zip
