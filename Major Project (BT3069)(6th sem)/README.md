# Multi-Disease Detection from Medical Images using Transfer Learning

## 📌 Project Overview

This project proposes a **Multi-Disease Detection (MDD) system** using Transfer Learning and Ensemble Methods for medical image classification.

Unlike traditional systems that focus on a single disease, our approach aims to **simultaneously detect multiple diseases across different domains**, including:

* 🧠 Brain (Alzheimer’s, Tumors)
* 🫁 Lung (COVID-19, Pneumonia)
* 👁️ Eye (Glaucoma, Diabetic Retinopathy)
* 🧴 Skin (Melanoma, Carcinoma)

---

## 🎯 Objective

To design a scalable AI-based framework that:

* Reduces dependency on large labeled datasets
* Improves diagnostic accuracy using ensemble learning
* Supports multi-domain disease detection

---

## 🧠 Methodology

### 🔹 1. Data Collection

Datasets used:

* NIH Chest X-ray14
* OASIS / ADNI (Brain MRI)
* HAM10000 (Skin)
* APTOS (Eye)

### 🔹 2. Preprocessing

* Image normalization
* Noise reduction (Gaussian blur)
* Data augmentation (rotation, flipping, zoom)

### 🔹 3. Transfer Learning

Pre-trained models used:

* EfficientNetB4
* ResNet50
* VGG16
* InceptionV3

### 🔹 4. Ensemble Learning

* Stacking and Soft Voting
* MRLA optimization (proposed)

---

## 📊 Key Features

* Multi-stage Transfer Learning pipeline
* Multi-disease classification framework
* Ensemble-based performance improvement
* Systematic literature review of 19 research papers

---

## 📈 Results

⚠️ Note: This project currently presents a **proposed framework** based on literature analysis. Experimental validation is part of future work.

Expected performance (based on literature):

* Accuracy: ~94%–99%
* AUROC: ~0.95+

---

## 🔍 Research Contribution

* Identifies limitations of single-disease models
* Proposes a unified multi-disease detection architecture
* Provides gap analysis (Explainable AI, OOD robustness, 3D imaging)

---

## 🚀 Future Work

* Implement full multi-disease model
* Integrate Explainable AI (Grad-CAM, LIME)
* Use self-supervised learning (SimCLR, MoCo)
* Extend to 3D medical imaging

---

## 👨‍💻 Team Members

* Gaurav Upadhyay
* Kritik Kumar
* Akshma Pundir
* Aman Kumar
* Ayush Sharma
* Himanshu Singh

---

## 📄 Research Paper

📎 [View Full Paper](paper/BT3069_Research%20Paper.pdf)

---

## ⚙️ Tech Stack

* Python
* TensorFlow / PyTorch
* OpenCV
* Google Colab

---

## 📌 Note

This repository contains the research and proposed architecture for academic purposes (B.Tech CSE 3rd year Major Project).
