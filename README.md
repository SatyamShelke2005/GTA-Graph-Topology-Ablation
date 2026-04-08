# GTA (Graph Topology Ablation) Challenge - MIT-WPU

This repository hosts the implementation and evaluation system for the **Graph Topology Ablation (GTA) challenge**, completed as part of the Data Science and Big Data Analytics coursework.

**Institution:** MIT - World Peace University (MIT-WPU), Pune  
**Department:** Department of Computer Science and Applications  
**Course:** Data Science and Big Data Analytics  

---

## 👥 Project Contributors
| Name | PRN | Role |
| :--- | :--- | :--- |
| **Satyam Anilrao Shelke** | 1132231165 | Lead Developer / Data Pipeline |
| **Harsha Purohit** | 1132231017 | Research & Model Validation |

---

## 🎯 Objective ##

Participants must generate predictions for two settings:

- ✅ **Ideal graph topology** – clean, unmodified node features.
- ✅ **Perturbed graph topology** – node features corrupted by a combination of distribution shift and Gaussian noise.

The goal is to build a classification system that is both accurate on clean data and robust to realistic feature corruptions.

### 🛠️ Technical Solution: The "Class Collapse" Fix
A critical challenge identified during development was **"Zero-Bias"**, where the model predicted only the majority class (0). 
- **The Fix:** Implemented **Dynamic Percentile Thresholding**. 
- **Mechanism:** Instead of a static $0.5$ decision boundary, the system calculates a threshold based on the top $35\%$-$40\%$ of softmax probabilities.
- **Result:** This ensures a diverse and accurate distribution of labels (1s and 0s) even under high entropy or noise.

---

## 📌 Dataset Description

For the validation phase, we utilized a protocol based on the **Iris Dataset** (Binary Classification) to verify pipeline integrity.

| Property | Value |
| :--- | :--- |
| **Task** | Binary graph/node classification |
| **Optimization** | Adam Optimizer ($lr=0.005$) |
| **Final Loss** | **0.026527** (NLL Training Loss) |
| **Architecture** | 3-Layer MLP with BatchNorm & Dropout (0.2) |

---

## 🚀 Getting Started

### Environment Setup
Create a Python virtual environment and install dependencies:
```powershell
.\venv\Scripts\activate